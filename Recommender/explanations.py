from Recommender.recommender import Recommender
from Recommender.group_aggregation_functions import get_group_agg_func  # NEW
from Recommender.diversification import load_genre_embedding, movie_similarity  # NEW

from rich import print
from rich.console import Console  # NEW
from rich.table import Table      # NEW

import numpy as np               # NEW
import pandas as pd


console = Console()  # NEW


def get_item_intensities(group: pd.DataFrame):
    """For each item returns how many users rated them."""
    rating_count = group.notnull().sum(axis=0)
    return rating_count


def get_user_intensities(group: pd.DataFrame):
    intensities = group.notnull().sum(axis=1)
    return intensities


def compute_popularity_distribution(item_intensities: pd.Series):
    distribution = item_intensities.value_counts()
    return distribution


# ---------- NEW: candidate selection based on genres + intensity ----------

def get_candidates(group_table: pd.DataFrame, item_id: int, top_k: int) -> pd.DataFrame:
    """
    Return up to `top_k` candidate explanation movies for `item_id`.

    Steps:
    1) Take movies that at least one user in the group rated (intensity > 0).
    2) Compute genre-based cosine similarity to `item_id`.
    3) Keep only movies with positive similarity.
    4) Sort primarily by intensity (how many users rated it), secondarily by similarity.
    """
    embeddings = load_genre_embedding()

    item_intensities = get_item_intensities(group_table)

    # Movies that the group has actually interacted with
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
            # We only keep positively similar items
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
    return df.head(top_k)


# ---------- NEW: rich helpers for nice printing ----------

def _load_movies_meta(path: str = "ml-latest-small/movies.csv") -> pd.DataFrame:
    movies = pd.read_csv(path)
    movies = movies.set_index("movieId")
    return movies


def print_user_intensities(group: pd.DataFrame):
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
):
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
):
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
):
    before = list(before_scores.sort_values(ascending=False).head(top_n).items())
    after = list(after_scores.sort_values(ascending=False).head(top_n).items())
    max_len = max(len(before), len(after))

    table = Table(
        title="Comparison of top-k aggregated rankings (before vs. after removal)",
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



def experiment_1():
    console.rule("[bold blue]Group CF explanation demo")

    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    movies_meta = _load_movies_meta()  # for pretty titles

    random_group_of_five = r.table.sample(5, random_state=42)
    print("\n[bold]Selected group (user IDs):[/bold]", list(random_group_of_five.index))
    print_user_intensities(random_group_of_five)

    console.rule("[bold green]Baseline group predictions and aggregated ranking")
    predictions_before = r.get_predictions_for_group(random_group_of_five)

    agg_func = get_group_agg_func("average")
    agg_before_df = agg_func(predictions_before)
    scores_before = agg_before_df.iloc[0].sort_values(ascending=False)

    print_aggregated_ranking(
        "Baseline aggregated ranking (top 10)",
        scores_before,
        movies_meta,
        top_n=10,
    )

    # Choose top-1 movie for the group
    target_movie_id = int(scores_before.index[0])
    print(
        f"\n[bold]Top-1 recommended movie for the group:[/bold] {target_movie_id} – "
        f"{movies_meta.loc[target_movie_id, 'title'] if target_movie_id in movies_meta.index else '<unknown>'}"
    )

    console.rule("[bold yellow]Finding candidate explanation items")
    candidates = get_candidates(
        group_table=random_group_of_five,
        item_id=target_movie_id,
        top_k=10,  # you can tweak this
    )

    if candidates.empty:
        print("[bold red]No suitable candidates found (no similar + rated movies).[/bold]")
        return

    print_candidates_table(target_movie_id, candidates, movies_meta)

    candidate_ids = list(candidates.index)

    console.rule("[bold red]Removing candidate items for the group and recomputing recommendations")

    base_table = r.table
    table_without_candidates = base_table.copy()
    table_without_candidates.loc[random_group_of_five.index, candidate_ids] = np.nan

    group_without_candidates = random_group_of_five.copy()
    group_without_candidates.loc[:, candidate_ids] = np.nan

    r_without = Recommender(table_without_candidates, sim_func="pearson")

    predictions_after = r_without.get_predictions_for_group(group_without_candidates)
    agg_after_df = agg_func(predictions_after)
    scores_after = agg_after_df.iloc[0].sort_values(ascending=False)

    print_aggregated_ranking(
        "Aggregated ranking AFTER removing candidate items (top 10)",
        scores_after,
        movies_meta,
        top_n=10,
    )

    print_ranking_comparison(scores_before, scores_after, movies_meta, top_n=10)

    print(
        "\n[bold]Summary:[/bold]\n"
        "- We computed group predictions and an aggregated ranking.\n"
        f"- Chosen target movie: [cyan]{target_movie_id}[/cyan].\n"
        "- We selected candidate explanation items based on genre similarity and how many group members rated them.\n"
        "- For those candidates, we removed their ratings for the 5 users (both in the full table used by CF and in the group view).\n"
        "- We recomputed the recommendations and showed how the top-k aggregated list changed."
    )


def experiment_2():
    """
    Experiment 2:
    - Take a random group of 5 users.
    - Compute CF predictions and aggregated ranking.
    - Pick the (second) top recommended movie for the group.
    - Find ALL movies that share at least one genre with this target movie.
      (and that at least one group member has rated)
    - Remove ALL ratings for these same-genre movies from the group users
      (both in the global rating table used by CF and in the group slice).
    - Recompute recommendations and show how the aggregated ranking changes.
    """
    console.rule("[bold blue]Group CF explanation demo – Experiment 2: Remove ALL same-genre items")

    # 1) Load recommender & movie metadata & genre embeddings
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    movies_meta = _load_movies_meta()
    embeddings = load_genre_embedding()

    # 2) Random group of 5
    random_group_of_five = r.table.sample(5, random_state=42)
    print("\n[bold]Selected group (user IDs):[/bold]", list(random_group_of_five.index))
    print_user_intensities(random_group_of_five)

    # 3) Baseline predictions and aggregated ranking
    console.rule("[bold green]Baseline group predictions and aggregated ranking")
    predictions_before = r.get_predictions_for_group(random_group_of_five)

    agg_func = get_group_agg_func("average")
    agg_before_df = agg_func(predictions_before)
    scores_before = agg_before_df.iloc[0].sort_values(ascending=False)

    print_aggregated_ranking(
        "Baseline aggregated ranking (top 10)",
        scores_before,
        movies_meta,
        top_n=10,
    )

    # Choose target movie (keeping your 'index[1]' choice from experiment_1)
    target_movie_id = int(scores_before.index[1])
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

    if target_movie_id not in embeddings.index:
        print("[bold red]Target movie does not have genre embedding – cannot run experiment_2.[/bold]")
        return

    # 4) Find ALL movies that share at least one genre with the target
    console.rule("[bold yellow]Finding ALL same-genre items rated by the group")

    target_vec = embeddings.loc[target_movie_id].to_numpy()

    # First: all movies that share at least one genre with target (sim > 0)
    same_genre_raw = []
    for mid in embeddings.index:
        if mid == target_movie_id:
            continue
        vb = embeddings.loc[mid].to_numpy()
        sim = float(np.dot(target_vec, vb))
        if sim > 0.0:
            same_genre_raw.append((int(mid), sim))

    if not same_genre_raw:
        print("[bold red]No movies share a genre with the target in the embeddings.[/bold]")
        return

    # Restrict to movies that the GROUP actually interacted with
    item_intensities = get_item_intensities(random_group_of_five)

    filtered = []
    for mid, sim in same_genre_raw:
        if mid in item_intensities.index and item_intensities[mid] > 0:
            filtered.append((mid, sim, item_intensities[mid]))

    if not filtered:
        print(
            "[bold yellow]There ARE same-genre movies, but none of them were rated by this group.[/bold]\n"
            "Removing them would not change anything for this group."
        )
        return

    # Build DataFrame: ALL same-genre items rated by the group
    same_genre_df = pd.DataFrame(
        {
            "similarity": {mid: sim for (mid, sim, cnt) in filtered},
            "intensity": {mid: cnt for (mid, sim, cnt) in filtered},
        }
    ).sort_values(by=["intensity", "similarity"], ascending=False)

    print_candidates_table(target_movie_id, same_genre_df, movies_meta)
    candidate_ids = list(same_genre_df.index)

    print(
        f"\n[bold]We will now remove ALL ratings for {len(candidate_ids)} movies[/bold] "
        f"that share at least one genre with the target and were rated by the group."
    )

    # 5) Show user intensities BEFORE removal
    console.rule("[bold green]User intensities BEFORE removal")
    print_user_intensities(random_group_of_five)

    # 6) Remove these items for the group users (both in the global table and group table)
    console.rule("[bold red]Removing same-genre items for the group and recomputing recommendations")

    base_table = r.table
    table_without_same_genre = base_table.copy()
    table_without_same_genre.loc[random_group_of_five.index, candidate_ids] = np.nan

    group_without_same_genre = random_group_of_five.copy()
    group_without_same_genre.loc[:, candidate_ids] = np.nan

    # Show user intensities AFTER removal
    console.rule("[bold green]User intensities AFTER removal")
    print_user_intensities(group_without_same_genre)

    # 7) New recommender on modified table, recompute predictions
    r_without = Recommender(table_without_same_genre, sim_func="pearson")

    predictions_after = r_without.get_predictions_for_group(group_without_same_genre)
    agg_after_df = agg_func(predictions_after)
    scores_after = agg_after_df.iloc[0].sort_values(ascending=False)

    print_aggregated_ranking(
        "Aggregated ranking AFTER removing ALL same-genre items (top 10)",
        scores_after,
        movies_meta,
        top_n=10,
    )

    # 8) Comparison before vs after
    print_ranking_comparison(scores_before, scores_after, movies_meta, top_n=10)

    # Small summary
    print(
        "\n[bold]Summary of experiment_2:[/bold]\n"
        f"- Target movie: [cyan]{target_movie_id} – {target_title}[/cyan] (genres: {target_genres}).\n"
        "- We identified ALL movies that share at least one genre with the target.\n"
        "- From those, we kept only movies that at least one user in the group had rated.\n"
        f"- This gave us {len(candidate_ids)} same-genre movies, with varying group intensities.\n"
        "- We removed ALL ratings for these movies for the 5 group users (simulating: they never liked any movie from that genre).\n"
        "- We recomputed CF predictions and the aggregated ranking and compared top-k before vs after.\n"
        "- This shows how strongly the whole genre cluster influences the recommendation of the target item."
    )

def experiment_3():
    """
    Experiment 3:
    - Take a random group of 5 users.
    - Compute CF predictions and aggregated ranking.
    - Pick a target recommended movie for the group.
    - Find ALL movies that share at least one genre with this target movie
      AND that at least one group member has rated.
    - For these same-genre movies, set ALL ratings of the 5 users to 0
      (dislike) instead of NaN (removal).
    - Recompute recommendations and show how the aggregated ranking changes.
    This corresponds to: "If you had really disliked all movies of this genre,
    how would the recommendations change?"
    """
    console.rule("[bold blue]Group CF explanation demo – Experiment 3: Set same-genre items to 0 (dislike)")

    # 1) Load recommender & movie metadata & genre embeddings
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    movies_meta = _load_movies_meta()
    embeddings = load_genre_embedding()

    # 2) Random group of 5
    random_group_of_five = r.table.sample(5, random_state=42)
    print("\n[bold]Selected group (user IDs):[/bold]", list(random_group_of_five.index))
    print_user_intensities(random_group_of_five)

    # 3) Baseline predictions and aggregated ranking
    console.rule("[bold green]Baseline group predictions and aggregated ranking")
    predictions_before = r.get_predictions_for_group(random_group_of_five)

    agg_func = get_group_agg_func("average")
    agg_before_df = agg_func(predictions_before)
    scores_before = agg_before_df.iloc[0].sort_values(ascending=False)

    print_aggregated_ranking(
        "Baseline aggregated ranking (top 10)",
        scores_before,
        movies_meta,
        top_n=10,
    )

    # Choose target movie (keeping your 'index[1]' choice)
    target_movie_id = int(scores_before.index[1])
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

    if target_movie_id not in embeddings.index:
        print("[bold red]Target movie does not have genre embedding – cannot run experiment_3.[/bold]")
        return

    # 4) Find ALL movies that share at least one genre with the target
    console.rule("[bold yellow]Finding ALL same-genre items rated by the group")

    target_vec = embeddings.loc[target_movie_id].to_numpy()

    # All movies with sim > 0 to target
    same_genre_raw = []
    for mid in embeddings.index:
        if mid == target_movie_id:
            continue
        vb = embeddings.loc[mid].to_numpy()
        sim = float(np.dot(target_vec, vb))
        if sim > 0.0:
            same_genre_raw.append((int(mid), sim))

    if not same_genre_raw:
        print("[bold red]No movies share a genre with the target in the embeddings.[/bold]")
        return

    # Restrict to movies that this GROUP actually interacted with
    item_intensities = get_item_intensities(random_group_of_five)

    filtered = []
    for mid, sim in same_genre_raw:
        if mid in item_intensities.index and item_intensities[mid] > 0:
            filtered.append((mid, sim, item_intensities[mid]))

    if not filtered:
        print(
            "[bold yellow]There ARE same-genre movies, but none of them were rated by this group.[/bold]\n"
            "Changing them to 0 would not affect this group."
        )
        return

    same_genre_df = pd.DataFrame(
        {
            "similarity": {mid: sim for (mid, sim, cnt) in filtered},
            "intensity": {mid: cnt for (mid, sim, cnt) in filtered},
        }
    ).sort_values(by=["intensity", "similarity"], ascending=False)

    print_candidates_table(target_movie_id, same_genre_df, movies_meta)
    candidate_ids = list(same_genre_df.index)

    print(
        f"\n[bold]We will now set ratings to 0 for {len(candidate_ids)} movies[/bold] "
        f"that share at least one genre with the target and were rated by the group.\n"
        "[bold]Interpretation:[/bold] 'If you had really disliked all movies of this genre...'"
    )

    # 5) User intensities BEFORE (they won't change much, but we show them for completeness)
    console.rule("[bold green]User intensities BEFORE setting dislikes")
    print_user_intensities(random_group_of_five)

    # 6) Set those items to 0 (dislike) for the group users
    console.rule("[bold red]Setting same-genre items to 0 for the group and recomputing recommendations")

    base_table = r.table
    table_with_dislikes = base_table.copy()
    group_with_dislikes = random_group_of_five.copy()

    # Only overwrite positions where there WAS a rating (keep NaNs as NaNs)
    mask_global = table_with_dislikes.loc[random_group_of_five.index, candidate_ids].notna()
    mask_group = group_with_dislikes.loc[:, candidate_ids].notna()

    table_with_dislikes.loc[random_group_of_five.index, candidate_ids] = (
        table_with_dislikes.loc[random_group_of_five.index, candidate_ids]
        .where(~mask_global, other=0.0)
    )
    group_with_dislikes.loc[:, candidate_ids] = (
        group_with_dislikes.loc[:, candidate_ids]
        .where(~mask_group, other=0.0)
    )

    console.rule("[bold green]User intensities AFTER setting dislikes (should be same, #rated doesn't change)")
    print_user_intensities(group_with_dislikes)

    # 7) New recommender on modified table, recompute predictions
    r_with_dislikes = Recommender(table_with_dislikes, sim_func="pearson")

    predictions_after = r_with_dislikes.get_predictions_for_group(group_with_dislikes)
    agg_after_df = agg_func(predictions_after)
    scores_after = agg_after_df.iloc[0].sort_values(ascending=False)

    print_aggregated_ranking(
        "Aggregated ranking AFTER setting same-genre items to 0 (top 10)",
        scores_after,
        movies_meta,
        top_n=10,
    )

    # 8) Comparison before vs after
    print_ranking_comparison(scores_before, scores_after, movies_meta, top_n=10)
# --- Extra: show which movies disappeared from the top-k after the change ---
    top_n = 10  # make sure this matches the value used above

    before_top = scores_before.sort_values(ascending=False).head(top_n)
    after_top = scores_after.sort_values(ascending=False).head(top_n)

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
            if mid_int in movies_meta.index:
                title_str = str(movies_meta.loc[mid_int, "title"])
            else:
                title_str = "<unknown>"

            table.add_row(str(mid_int), title_str, f"{old_score:.3f}")

        console.print(table)
    else:
        print(
            "[bold yellow]No movies disappeared from the top-k:[/bold] "
            "all items from the old top-k are still present in the new top-k."
        )

    print(
        "\n[bold]Summary of experiment_3:[/bold]\n"
        f"- Target movie: [cyan]{target_movie_id} – {target_title}[/cyan] (genres: {target_genres}).\n"
        "- We identified ALL movies that share at least one genre with the target.\n"
        "- From those, we kept only movies that at least one user in the group had rated.\n"
        f"- This gave us {len(candidate_ids)} same-genre movies.\n"
        "- For those movies, we changed existing ratings of the 5 group users to 0 (dislike), "
        "instead of removing them.\n"
        "- We recomputed CF predictions and the aggregated ranking and compared top-k before vs after.\n"
        "- This corresponds to the counterfactual: 'If you had really disliked all movies of this genre, "
        "this is how your recommendations would change.'"
    )

if __name__ == "__main__":
    # experiment_1()
    experiment_3()
