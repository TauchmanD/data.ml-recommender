import numpy as np
import pandas as pd


# def get_satisfaction_for_users(
#     predictions: pd.DataFrame, aggregated_rankings: pd.DataFrame
# ):
#     """Computes satisfaction for user"""
#     print("aggregated: ", aggregated_rankings)
#     movie_ids = aggregated_rankings.columns
#     print("Movie ids: " , movie_ids)
#     k = len(movie_ids)
#
#     group_satisfactions = pd.DataFrame(columns=["userId", "satisfaction"]).set_index(
#         "userId"
#     )
#     for index_user, row in predictions.iterrows():
#         group_satisfactions.loc[index_user] = sum(row.loc[movie_ids])
#
#     ideal_satisfaction = pd.DataFrame(columns=["userId", "satisfaction"]).set_index(
#         "userId"
#     )
#
#     for index_user, row in predictions.iterrows():
#         top_k_movies = row.sort_values(ascending=False)[
#             :k
#         ]  # We assume that there are no NaNs (well synce it is output from Collaborative Filtering there will not be any)
#         ideal_satisfaction.loc[index_user] = sum(top_k_movies)
#
#     satisfactions = group_satisfactions / ideal_satisfaction
#     satisfactions = satisfactions.replace([np.inf, -np.inf], np.nan).fillna(0.0)
#     return satisfactions



def get_satisfaction_for_users(predictions: pd.DataFrame, aggregated_rankings: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    predictions: users x movieId (CF scores). Index=userId, columns=movieId.
    aggregated_rankings:
      - the original 1×K DataFrame from your aggregator (row name e.g. 'mean', columns=movieIds), or
      - several such rows concatenated (rows = rounds, columns = union of movieIds).
    Returns: users x rounds DataFrame of satisfaction (group_sum / ideal_top_k_sum).
             Column names are the row labels from aggregated_rankings (e.g., 'mean', 'least_misery', ...).
    """
    # Normalize to DataFrame
    # agg = aggregated_rankings.to_frame().T if isinstance(aggregated_rankings, pd.Series) else aggregated_rankings.copy()
    agg = aggregated_rankings.copy()

    # Build a membership indicator: rounds x movieId (1 if the round includes the movie)
    # - We only care about *presence* of a column per round. Missing columns become NaN after reindex → False.
    indicator = agg.reindex(columns=predictions.columns).notna().astype(int)

    # Group satisfaction: sum of user predictions over the round's movies
    # users x movieId  ·  movieId x rounds  -> users x rounds
    group_satisfactions = predictions.dot(indicator.T)

    # Ideal satisfaction: for each user & round, sum top-k predictions (k = #movies in that round)
    k_per_round = indicator.sum(axis=1).to_list()

    vals = predictions.to_numpy()
    vals = np.nan_to_num(vals, nan=0.0)
    order = np.argsort(-vals, axis=1)                  # sort each user row desc
    sorted_vals = np.take_along_axis(vals, order, axis=1)
    csum = np.cumsum(sorted_vals, axis=1)

    n_users = predictions.shape[0]
    ideal = np.zeros((n_users, len(k_per_round)), dtype=float)
    for j, k in enumerate(k_per_round):
        if k > 0:
            k_eff = min(k, csum.shape[1])
            ideal[:, j] = csum[:, k_eff - 1]          # top-k sum per user

    ideal_satisfaction = pd.DataFrame(ideal, index=predictions.index, columns=indicator.index)

    # Final ratio, cleaned
    out = group_satisfactions.div(ideal_satisfaction)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

def get_overall_satisfaction(predictions, sequence):
    satisfactions = get_satisfaction_for_users(predictions, sequence)
    # for idx, seq in sequence.iterrows():
    #     print("_"*10)
    #     print(seq)
    #     print("_"*10)
    #     satisfactions.loc[idx] = get_satisfaction_for_users(predictions, seq)
    # print("Those are satisfactions per the rounds")
    # print(satisfactions)
    return satisfactions.mean(axis=1)
    
    

def get_disagreements_for_users(satisfactions):
    disagreement = satisfactions.max() - satisfactions.min()
    return disagreement.item()

def get_disagreements_per_user(satisfactions):
    max_satisfaction = satisfactions.max()
    disagreements = max_satisfaction - satisfactions
    return disagreements

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


def test_user_disagreements():
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
    satisfactions = get_satisfaction_for_users(predictions,aggregated_rankings)
    print(satisfactions)
    disagreements = get_disagreements_per_user(satisfactions)
    print(disagreements)

def test_overall_satisfaction():
    from group_aggregation_functions import get_group_agg_func
    agg_func = get_group_agg_func("average")
    least_agg_func = get_group_agg_func("least_misery")
    predictions = pd.DataFrame(
        [
            [1, 1, 0.8],
            [1, 2, 0.1],
            [1, 3, 0.4],
            [1, 4, 0.9],
            [2, 1, 0.3],
            [2, 2, 0.9],
            [2, 3, 100.1],
            [2, 4, 0.1],
            [3, 1, 0.8],
            [3, 2, 0.4],
            [3, 3, 0.1],
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
    least_misery_rankings = least_agg_func(predictions).iloc[:,:2]
    print("AVERAGE")
    print(aggregated_rankings)
    print("LEAST MISERY")
    print(least_misery_rankings)
    
# Average satisfaction
#         satisfaction
# userId
# 1           0.705882
# 2           0.994059
# 3           0.750000
# 4           0.866667
# Least Misery satisfaction
#         satisfaction
# userId
# 1           0.529412
# 2           0.011881
# 3           1.000000
# 4           0.533333
    average_satisfactions = get_satisfaction_for_users(predictions,aggregated_rankings)
    least_misery_satisfactions = get_satisfaction_for_users(predictions,least_misery_rankings)

    print("Average satisfaction")
    print(average_satisfactions)
    print("Least Misery satisfaction")
    print(least_misery_satisfactions )

    
    sequence_example = pd.concat([aggregated_rankings, least_misery_rankings])
    print("Sequence: ")
    print(sequence_example)


    print("GEtting simple satisfactions")
    satisfactions = get_satisfaction_for_users(predictions,aggregated_rankings)
    print("_"*10)
    print(satisfactions)
    disagreements = get_disagreements_per_user(satisfactions)
    print(disagreements)
    print("getting not simple: ")
    overall_satisfactions = get_overall_satisfaction(predictions,sequence_example)
    print("OVerall: ")
    print(overall_satisfactions)

if __name__ == "__main__":
    # example_for_presentation()
    test_overall_satisfaction()
