from Recommender.Explanation.candidates import get_candidates, get_movie_name_from_id
from Recommender.recommender import Recommender
from Recommender.group_aggregation_functions import get_group_agg_func

from rich import print
import numpy as np
import pandas as pd


def _recompute_aggregated_list(
    recommender: Recommender,
    group_ratings: pd.DataFrame,
    top_k: int | None = None,
) -> pd.Series:
    """
    Helper: run CF for a group, aggregate, return sorted Series (movieId -> score).
    """
    predictions = recommender.get_predictions_for_group(group_ratings)
    agg_func = get_group_agg_func("average")  # assuming average aggregation
    agg_df = agg_func(predictions)
    scores = agg_df.iloc[0].sort_values(ascending=False)
    if top_k is not None:
        return scores.head(top_k)
    return scores


def get_explanation_for_a_group(
    original_group_ratings: pd.DataFrame,
    aggregated_list: pd.Series,
) -> dict[int, list[str]]:
    """
    This method will call all the objects from the get_candidates() method.
    For each candidate generator:

    1. It generates a set of candidate movies (e.g. by decade, by agreement, by genre).
    2. It simulates a counterfactual world where the group did NOT rate those candidates
       (i.e. their ratings are set to NaN for group users).
    3. It recomputes group recommendations and aggregated ranking.
    4. It compares the new aggregated list with the original aggregated_list.
    5. For each movie that was in the original aggregated_list but disappears from
       the new top-k, it records an explanation using candidate_obj.get_explanation_string(movie_id).

    Result:
        For each movie in aggregated_list there will be zero or more explanations.
        Returned as: { movie_id: [explanation_str1, explanation_str2, ...], ... }
    """
    # We assume aggregated_list is already the baseline ranking for this group.
    baseline_movies = list(aggregated_list.index)
    top_k = len(baseline_movies)

    # Prepare result structure
    explanations: dict[int, list[str]] = {int(mid): [] for mid in baseline_movies}

    # Load full recommender and base rating table
    base_recommender = Recommender.load_from_path("ml-latest-small/ratings.csv")
    base_table = base_recommender.table

    group_user_ids = original_group_ratings.index

    # Ensure we have the same users in the base table
    # (original_group_ratings should be a subset of base_table rows)
    missing_users = [u for u in group_user_ids if u not in base_table.index]
    if missing_users:
        raise ValueError(f"Some group users not found in base rating table: {missing_users}")

    # Iterate over all candidate generators
    for cand in get_candidates():
        print(f"\n[bold blue]Running candidate generator:[/bold blue] {cand.__class__.__name__}")

        # 1) Generate candidate movie ids from original group ratings
        candidate_ids = cand(original_group_ratings)
        candidate_ids = pd.Index(candidate_ids).astype(int)

        if candidate_ids.empty:
            print("  [yellow]No candidates found for this generator.[/yellow]")
            continue

        # Filter to columns that actually exist in the base table
        candidate_ids_in_table = [m for m in candidate_ids if m in base_table.columns]
        if not candidate_ids_in_table:
            print("  [yellow]No candidate ids appear in the rating table.[/yellow]")
            continue

        print(f"  Candidate set size: {len(candidate_ids_in_table)}")

        # 2) Create a modified rating table where the group "did not rate" these candidates
        modified_table = base_table.copy()
        modified_table.loc[group_user_ids, candidate_ids_in_table] = np.nan
        # modified_table.loc[group_user_ids, candidate_ids_in_table] = 0

        # The group view in this counterfactual world
        modified_group_ratings = modified_table.loc[group_user_ids]

        # 3) Recompute recommendations & aggregated list
        modified_recommender = Recommender(modified_table, sim_func="pearson")
        new_scores = _recompute_aggregated_list(
            modified_recommender,
            modified_group_ratings,
            top_k=top_k,
        )

        new_top_movies = set(new_scores.index.astype(int))

        # 4) Movies that disappeared from top-k due to this candidate removal
        for mid in baseline_movies:
            mid_int = int(mid)
            if mid_int not in new_top_movies:
                explanation_str = cand.get_explanation_string(mid_int)
                explanations[mid_int].append(explanation_str)

    return explanations


def generate_explanations_experiment(
    group_size: int = 5,
    top_k: int = 10,
    random_state: int = 42,
):
    """
    Full pipeline experiment:

    1. Create a random group of `group_size` users.
    2. Generate recommendations for the group (top_k).
    3. Run the explanation pipeline.
    4. Print out the explanations for each movie in the list.
    """
    print("[bold green]=== Generating explanations experiment ===[/bold green]")

    # 1) Load recommender and pick random group
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    group = r.table.sample(group_size, random_state=random_state)

    print("\n[bold]Selected group (user IDs):[/bold]", list(group.index))

    # 2) Baseline aggregated list
    agg_func = get_group_agg_func("average")
    predictions = r.get_predictions_for_group(group)
    agg_df = agg_func(predictions)
    aggregated_list = agg_df.iloc[0].sort_values(ascending=False).head(top_k)

    print("\n[bold magenta]Baseline group top-k movies:[/bold magenta]")
    for rank, (mid, score) in enumerate(aggregated_list.items(), start=1):
        name = get_movie_name_from_id(mid)
        print(f"  {rank:2d}. [{mid}] '{name}'  (score={score:.3f})")

    # 3) Run explanation pipeline
    explanations = get_explanation_for_a_group(group, aggregated_list)

    # 4) Print explanations per movie
    print("\n[bold cyan]Explanations per movie:[/bold cyan]")
    for rank, (mid, score) in enumerate(aggregated_list.items(), start=1):
        mid_int = int(mid)
        movie_name = get_movie_name_from_id(mid_int)
        movie_exps = explanations.get(mid_int, [])

        print(f"\n[bold]{rank}. [{mid_int}] '{movie_name}' (score={score:.3f})[/bold]")

        if not movie_exps:
            print("   [yellow]No counterfactual explanations found for this movie.[/yellow]")
        else:
            for idx, exp in enumerate(movie_exps, start=1):
                print(f"   {idx}) {exp}")


def main():
    generate_explanations_experiment()


if __name__ == "__main__":
    main()
