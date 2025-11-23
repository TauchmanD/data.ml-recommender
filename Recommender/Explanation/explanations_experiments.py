from Recommender.recommender import Recommender
from Recommender.group_aggregation_functions import get_group_agg_func
from Recommender.diversification import load_genre_embedding, movie_similarity

from rich import print
from rich.console import Console
from rich.table import Table

import numpy as np
import pandas as pd


console = Console()


# ---------- basic stats ----------

def get_item_intensities(group: pd.DataFrame) -> pd.Series:
    """For each item returns how many users rated them."""
    rating_count = group.notnull().sum(axis=0)
    return rating_count


def get_user_intensities(group: pd.DataFrame) -> pd.Series:
    intensities = group.notnull().sum(axis=1)
    return intensities


def compute_popularity_distribution(item_intensities: pd.Series) -> pd.Series:
    distribution = item_intensities.value_counts()
    return distribution


# ---------- candidate selection based on genres + intensity ----------

def get_candidates(group_table: pd.DataFrame, item_id: int, top_k: int = 10) -> pd.DataFrame:
    """
    Return candidate explanation movies for `item_id`.

    Steps:
    1) Take movies that at least one user in the group rated (intensity > 0).
    2) Compute genre-based cosine similarity to `item_id` (using embeddings).
    3) Keep only movies with positive similarity.
    4) Sort primarily by intensity (# group members who rated it), secondarily by similarity.
    5) If top_k > 0 → return top_k; if top_k == 0 → return ALL such movies.

    Returns a DataFrame indexed by movieId with columns:
    - "similarity"
    - "intensity"
    """
    embeddings = load_genre_embedding()

    if item_id not in embeddings.index:
        return pd.DataFrame(columns=["similarity", "intensity"])

    item_intensities = get_item_intensities(group_table)

    # Movies that the group has actually interacted with and that have an embedding
    candidate_ids = [
        m_id
        for m_id, cnt in item_intensities.items()
        if cnt > 0 and m_id in embeddings.index and m_id != item_id
    ]

    if not candidate_ids:
        return pd.DataFrame(columns=["similarity", "intensity"])

    sims = {}
    ints = {}

    for m in candidate_ids:
        sim = movie_similarity(int(item_id), int(m), embeddings)
        if sim <= 0.0:
            # Keep only positively similar items (share at least one genre)
            continue
        sims[m] = sim
        ints[m] = item_intensities[m]

    if not sims:
        return pd.DataFrame(columns=["similarity", "intensity"])

    df = pd.DataFrame(
        {
            "similarity": pd.Series(sims),
            "intensity": pd.Series(ints),
        }
    )

    df = df.sort_values(by=["intensity", "similarity"], ascending=False)

    if top_k and top_k > 0:
        return df.head(top_k)
    return df


# ---------- rich helpers for nice printing ----------

def _load_movies_meta(path: str = "ml-latest-small/movies.csv") -> pd.DataFrame:
    movies = pd.read_csv(path)
    movies = movies.set_index("movieId")
    return movies


def print_user_intensities(group: pd.DataFrame) -> None:
    intensities = get_user_intensities(group)

    table = Table(
        title="User rating intensities in group",
        show_lines=True,
        title_justify="left",
    )
    table.add_column("User ID", justify="right", style="bold cyan")
    table.add_column("# rated items", justify="right")

    for user_id, cnt in intensities.items():
        table.add_row(str(user_id), str(int(cnt)))

    console.print(table)


def print_aggregated_ranking(
    title: str,
    scores: pd.Series,
    movies: pd.DataFrame,
    top_n: int = 10,
) -> None:
    table = Table(
        title=title,
        show_lines=True,
        title_justify="left",
    )
    table.add_column("Rank", justify="right", style="bold magenta")
    table.add_column("Movie ID", justify="right")
    table.add_column("Title")
    table.add_column("Aggregated score", justify="right")

    for rank, (movie_id, score) in enumerate(scores.head(top_n).items(), start=1):
        movie_id_int = int(movie_id)
        if movie_id_int in movies.index:
            title_str = str(movies.loc[movie_id_int, "title"])
        else:
            title_str = "<unknown>"

        table.add_row(
            str(rank),
            str(movie_id_int),
            title_str,
            f"{float(score):.3f}",
        )

    console.print(table)


def print_candidates_table(
    target_movie_id: int,
    candidates: pd.DataFrame,
    movies: pd.DataFrame,
) -> None:
    table = Table(
        title=f"Candidate explanation items for movie {target_movie_id}",
        show_lines=True,
        title_justify="left",
    )
    table.add_column("Movie ID", justify="right")
    table.add_column("Title")
    table.add_column("Similarity (genres)", justify="right")
    table.add_column("# users in group", justify="right")

    for movie_id, row in candidates.iterrows():
        movie_id_int = int(movie_id)
        if movie_id_int in movies.index:
            title_str = str(movies.loc[movie_id_int, "title"])
        else:
            title_str = "<unknown>"

        table.add_row(
            str(movie_id_int),
            title_str,
            f"{row['similarity']:.3f}",
            str(int(row["intensity"])),
        )

    console.print(table)


def print_ranking_comparison(
    before_scores: pd.Series,
    after_scores: pd.Series,
    movies: pd.DataFrame,
    top_n: int = 10,
) -> None:
    before = list(before_scores.sort_values(ascending=False).head(top_n).items())
    after = list(after_scores.sort_values(ascending=False).head(top_n).items())
    max_len = max(len(before), len(after))

    table = Table(
        title="Comparison of top-k aggregated rankings (before vs. after change)",
        show_lines=True,
        title_justify="left",
    )
    table.add_column("Rank", justify="right", style="bold magenta")
    table.add_column("Before: Movie (score)")
    table.add_column("After: Movie (score)")

    def fmt(movie_id, score):
        movie_id_int = int(movie_id)
        if movie_id_int in movies.index:
            title_str = movies.loc[movie_id_int, "title"]
        else:
            title_str = "<unknown>"
        return f"[bold]{movie_id_int}[/bold] – {title_str}  ([cyan]{score:.3f}[/cyan])"

    for i in range(max_len):
        rank_str = str(i + 1)
        if i < len(before):
            b_id, b_score = before[i]
            before_str = fmt(b_id, float(b_score))
        else:
            before_str = ""

        if i < len(after):
            a_id, a_score = after[i]
            after_str = fmt(a_id, float(a_score))
        else:
            after_str = ""

        table.add_row(rank_str, before_str, after_str)

    console.print(table)


def print_disappeared_from_topk(
    before_scores: pd.Series,
    after_scores: pd.Series,
    movies: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """Show which movies disappeared from the top-k after the change."""
    before_top = before_scores.sort_values(ascending=False).head(top_n)
    after_top = after_scores.sort_values(ascending=False).head(top_n)

    missing_ids = [m for m in before_top.index if m not in after_top.index]

    if missing_ids:
        table = Table(
            title="Movies that were in previous top-k but NOT in new top-k",
            show_lines=True,
            title_justify="left",
        )
        table.add_column("Movie ID", justify="right")
        table.add_column("Title")
        table.add_column("Old score", justify="right")

        for mid in missing_ids:
            mid_int = int(mid)
            old_score = float(before_top[mid])
            if mid_int in movies.index:
                title_str = str(movies.loc[mid_int, "title"])
            else:
                title_str = "<unknown>"

            table.add_row(str(mid_int), title_str, f"{old_score:.3f}")

        console.print(table)
    else:
        print(
            "[bold yellow]No movies disappeared from the top-k:[/bold] "
            "all items from the old top-k are still present in the new top-k."
        )


# ---------- core mutation helper ----------

def apply_set_value_to_candidates(
    base_table: pd.DataFrame,
    group_table: pd.DataFrame,
    group_user_ids: pd.Index,
    candidate_ids: list[int],
    set_value,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (new_base_table, new_group_table) where all candidate_ids for
    users in group_user_ids are modified according to set_value:

    - If set_value is NaN -> those ratings are REMOVED (set to NaN).
    - Otherwise -> existing ratings (non-NaN) are overwritten with set_value,
      NaNs remain NaNs.
    """
    new_base = base_table.copy()
    new_group = group_table.copy()

    if pd.isna(set_value):
        # Remove ratings (set to NaN) regardless of their current value
        new_base.loc[group_user_ids, candidate_ids] = np.nan
        new_group.loc[:, candidate_ids] = np.nan
    else:
        # Overwrite only where there WAS a rating (keep NaNs as NaNs)
        mask_global = new_base.loc[group_user_ids, candidate_ids].notna()
        mask_group = new_group.loc[:, candidate_ids].notna()

        new_base.loc[group_user_ids, candidate_ids] = (
            new_base.loc[group_user_ids, candidate_ids]
            .where(~mask_global, other=set_value)
        )
        new_group.loc[:, candidate_ids] = (
            new_group.loc[:, candidate_ids]
            .where(~mask_group, other=set_value)
        )

    return new_base, new_group


# ---------- generic experiment runner ----------

def run_experiment(
    experiment_name: str,
    target_rank: int,
    top_k_candidates: int,
    set_value,
    operation_label: str,
    group_size: int = 5,
    random_state: int = 42,
    top_n_display: int = 10,
) -> None:
    """
    Generic pipeline:

    - Sample a random group of `group_size` users.
    - Compute baseline CF predictions & aggregated ranking.
    - Pick target movie at position `target_rank` in the aggregated ranking.
    - Find candidate movies using `get_candidates`:
        top_k_candidates > 0 -> only top_k candidates (by intensity)
        top_k_candidates == 0 -> ALL same-genre candidates rated by the group
    - Modify the group's ratings for these candidates using `set_value`.
    - Recompute CF and show before/after rankings and which movies disappeared from top-k.
    """
    console.rule(f"[bold blue]{experiment_name}")

    # 1) Load data
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    movies_meta = _load_movies_meta()

    # Random group
    group = r.table.sample(group_size, random_state=random_state)
    print("\n[bold]Selected group (user IDs):[/bold]", list(group.index))
    print_user_intensities(group)

    # 2) Baseline predictions and aggregated ranking
    console.rule("[bold green]Baseline group predictions and aggregated ranking")
    predictions_before = r.get_predictions_for_group(group)

    agg_func = get_group_agg_func("average")
    agg_before_df = agg_func(predictions_before)
    scores_before = agg_before_df.iloc[0].sort_values(ascending=False)

    print_aggregated_ranking(
        "Baseline aggregated ranking (top 10)",
        scores_before,
        movies_meta,
        top_n=top_n_display,
    )

    # 3) Choose target movie by rank in aggregated list
    if target_rank >= len(scores_before):
        target_rank = 0  # fallback to best if something weird happens

    target_movie_id = int(scores_before.index[target_rank])
    target_title = (
        movies_meta.loc[target_movie_id, "title"]
        if target_movie_id in movies_meta.index
        else "<unknown>"
    )
    target_genres = (
        movies_meta.loc[target_movie_id, "genres"]
        if "genres" in movies_meta.columns and target_movie_id in movies_meta.index
        else "<unknown genres>"
    )

    print(
        f"\n[bold]Target movie for explanation:[/bold] {target_movie_id} – {target_title}\n"
        f"[bold]Genres:[/bold] {target_genres}"
    )

    # 4) Candidate selection
    console.rule("[bold yellow]Finding candidate explanation items")

    candidates = get_candidates(
        group_table=group,
        item_id=target_movie_id,
        top_k=top_k_candidates,
    )

    if candidates.empty:
        print("[bold red]No suitable candidates found (no similar + rated movies).[/bold]")
        return

    print_candidates_table(target_movie_id, candidates, movies_meta)
    candidate_ids = list(candidates.index)

    set_value_str = "NaN (removed)" if pd.isna(set_value) else str(set_value)
    print(
        f"\n[bold]We will now apply the following change to {len(candidate_ids)} candidate movies:[/bold]\n"
        f"Operation: {operation_label} (set value = {set_value_str})"
    )

    # 5) Intensities BEFORE
    console.rule("[bold green]User intensities BEFORE change")
    print_user_intensities(group)

    # 6) Apply change to candidates
    console.rule("[bold red]Applying change to candidate items and recomputing recommendations")

    base_table = r.table
    new_base, new_group = apply_set_value_to_candidates(
        base_table=base_table,
        group_table=group,
        group_user_ids=group.index,
        candidate_ids=candidate_ids,
        set_value=set_value,
    )

    console.rule("[bold green]User intensities AFTER change")
    print_user_intensities(new_group)

    # 7) Recompute CF & aggregated ranking
    r_modified = Recommender(new_base, sim_func="pearson")

    predictions_after = r_modified.get_predictions_for_group(new_group)
    agg_after_df = agg_func(predictions_after)
    scores_after = agg_after_df.iloc[0].sort_values(ascending=False)

    print_aggregated_ranking(
        "Aggregated ranking AFTER change (top 10)",
        scores_after,
        movies_meta,
        top_n=top_n_display,
    )

    # 8) Comparison + disappeared items
    print_ranking_comparison(scores_before, scores_after, movies_meta, top_n=top_n_display)
    print_disappeared_from_topk(scores_before, scores_after, movies_meta, top_n=top_n_display)

    # 9) Summary
    print(
        f"\n[bold]Summary of {experiment_name}:[/bold]\n"
        f"- Target movie: [cyan]{target_movie_id} – {target_title}[/cyan] (genres: {target_genres}).\n"
        "- We identified movies sharing at least one genre with the target and rated by the group.\n"
        f"- Candidate selection mode: top_k_candidates = {top_k_candidates} "
        f"({'ALL same-genre items' if top_k_candidates == 0 else 'top items by popularity (intensity)' }).\n"
        f"- We then {operation_label} for these movies for the group users (set value = {set_value_str}).\n"
        "- Finally, we recomputed CF predictions and aggregated rankings and examined how the top-k list changed."
    )


# ---------- concrete experiments built on top of run_experiment ----------

def experiment_1():
    """
    Experiment 1:
    - Target = top-1 movie (rank 0).
    - Candidates = only top-k similar & popular items (top_k_candidates=10).
    - Operation = remove ratings (set to NaN).
    """
    run_experiment(
        experiment_name="Experiment 1: Remove TOP similar items",
        target_rank=0,
        top_k_candidates=10,
        set_value=np.nan,
        operation_label="removed ratings for candidate movies (set to NaN)",
    )


def experiment_2():
    """
    Experiment 2:
    - Target = 2nd ranked movie (rank 1).
    - Candidates = ALL same-genre items (top_k_candidates=0).
    - Operation = remove ratings (set to NaN).
    """
    run_experiment(
        experiment_name="Experiment 2: Remove ALL same-genre items",
        target_rank=0,
        top_k_candidates=0,  # 0 => ALL same-genre items
        set_value=np.nan,
        operation_label="removed ALL ratings for same-genre movies (set to NaN)",
    )


def experiment_3():
    """
    Experiment 3:
    - Target = 2nd ranked movie (rank 1).
    - Candidates = ALL same-genre items (top_k_candidates=0).
    - Operation = set ratings to 0 (strong dislike).
    """
    run_experiment(
        experiment_name="Experiment 3: Set same-genre items to 0 (dislike)",
        target_rank=1,
        top_k_candidates=0,  # 0 => ALL same-genre items
        set_value=0.0,
        operation_label="changed ratings for same-genre movies to 0 (dislike)",
    )


if __name__ == "__main__":
    # experiment_1()
    experiment_2()
    # experiment_3()
