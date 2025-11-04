import numpy as np
import pandas as pd


def get_satisfaction_for_users(
    predictions: pd.DataFrame, aggregated_rankings: pd.DataFrame
):
    """Computes satisfaction for user"""
    movie_ids = aggregated_rankings.columns
    k = len(movie_ids)
    group_satisfactions = pd.DataFrame(columns=["userId", "satisfaction"]).set_index(
        "userId"
    )
    for index_user, row in predictions.iterrows():
        group_satisfactions.loc[index_user] = sum(row.loc[movie_ids])

    ideal_satisfaction = pd.DataFrame(columns=["userId", "satisfaction"]).set_index(
        "userId"
    )

    for index_user, row in predictions.iterrows():
        top_k_movies = row.sort_values(ascending=False)[
            :k
        ]  # We assume that there are no NaNs (well synce it is output from Collaborative Filtering there will not be any)
        ideal_satisfaction.loc[index_user] = sum(top_k_movies)

    satisfactions = group_satisfactions / ideal_satisfaction
    satisfactions = satisfactions.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return satisfactions


def get_disagreements_for_users(satisfactions):
    disagreement = satisfactions.max() - satisfactions.min()
    return disagreement.item()


def get_disagreements_based_on_order(predictions, aggregated_rankings):

    if isinstance(aggregated_rankings, pd.DataFrame):
        group_items = list(aggregated_rankings.columns)
    else:
        group_items = list(aggregated_rankings)

    # keep only items present in predictions
    group_items = [c for c in group_items if c in predictions.columns]
    K = len(group_items)
    if K == 0:
        per_user = pd.Series(0.0, index=predictions.index, name="disagreement")
        return per_user, 0.0

    # per-user ranks (1 = best)
    ranks = predictions.rank(axis=1, ascending=False, method="first")

    # penalties for the K group items
    penalties = (ranks[group_items] - K).clip(lower=0)

    per_user = penalties.sum(axis=1).rename("disagreement")
    group_total = float(per_user.sum())
    return penalties, group_total


def test():
    from group_aggregation_functions import get_group_agg_func

    agg_func = get_group_agg_func("average")
    predictions = pd.DataFrame(
        [
            [1, 1, 0.8],
            [1, 2, 0.4],
            [1, 3, 0.1],
            [1, 4, 0.7],
            [2, 1, 0.3],
            [2, 2, 0.5],
            [2, 3, 0.1],
            [2, 4, 0.1],
            [3, 1, 0.8],
            [3, 2, 0.4],
            [3, 3, 0.4],
            [3, 4, 0.1],
            [4, 1, 0.3],
            [4, 2, 0.5],
            [4, 3, 1.0],
            [4, 4, 0.1],
        ],
        columns=["userId", "movieId", "rating"],
    )
    predictions = predictions.pivot_table(
        index="userId", columns="movieId", values="rating"
    )
    aggregated_rankings = agg_func(predictions).iloc[:, :2]
    print(predictions)
    print("Aggregated")
    print(aggregated_rankings)
    print("_" * 10)

    print(predictions)
    print(predictions.reset_index())
    print("DISAGREEMENTS")
    penalties, group_disagreement = get_disagreements_based_on_order(
        predictions, aggregated_rankings
    )
    print(penalties, group_disagreement)
    print("Group disagreement is: ", group_disagreement)
    reorder = get_group_agg_func("reorder")
    aggregation = reorder(predictions, aggregated_rankings, penalties)
    print("new aggregation")
    print(aggregation)
    print("_" * 10)
    print("NEW DISAGREEMENTS")
    penalties, group_disagreement = get_disagreements_based_on_order(
        predictions, aggregated_rankings
    )
    print(penalties, group_disagreement)
    print("Group disagreement is: ", group_disagreement)
    aggregation = reorder(predictions, aggregated_rankings, penalties)
    print("new aggregation")
    print(aggregation)
    print("_" * 10)
    print("NEW DISAGREEMENTS")
    penalties, group_disagreement = get_disagreements_based_on_order(
        predictions, aggregated_rankings
    )
    print(penalties, group_disagreement)
    print("Group disagreement is: ", group_disagreement)


def example_for_presentation():
    from group_aggregation_functions import get_group_agg_func

    agg_func = get_group_agg_func("average")
    predictions = pd.DataFrame(
        [
            [1, 1, 0.8],
            [1, 2, 0.1],
            [1, 3, 0.4],
            [1, 4, 0.9],
            [2, 1, 0.3],
            [2, 2, 0.9],
            [2, 3, 0.1],
            [2, 4, 0.1],
            [3, 1, 0.8],
            [3, 2, 0.4],
            [3, 3, 0.4],
            [3, 4, 0.1],
            [4, 1, 0.3],
            [4, 2, 0.5],
            [4, 3, 1.0],
            [4, 4, 0.1],
        ],
        columns=["userId", "movieId", "rating"],
    )
    predictions = predictions.pivot_table(
        index="userId", columns="movieId", values="rating"
    )
    aggregated_rankings = agg_func(predictions).iloc[:, :2]
    print(predictions)
    print("Aggregated")
    print(aggregated_rankings)
    print("_" * 10)

    print(predictions)
    print(predictions.reset_index())
    print("DISAGREEMENTS")
    penalties, group_disagreement = get_disagreements_based_on_order(
        predictions, aggregated_rankings
    )
    print(penalties, group_disagreement)


if __name__ == "__main__":
    example_for_presentation()
