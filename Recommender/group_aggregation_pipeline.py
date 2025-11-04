from recommender import Recommender
import pandas as pd
import random
import time
from group_aggregation_functions import get_group_agg_func
from rich import print
from rich.table import Table
from rich.console import Console


def test_that_CF_are_same():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    value = r.predict(3, 3)

    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    single_user_table = r.table.loc[[3]]
    predictions = r.get_predictions_for_group(single_user_table)
    prediction_movie_3 = predictions.loc[3, 3]
    print("predictions: \n ", predictions)

    result = value == prediction_movie_3
    print(result)
    return result


def aggregate_with_average():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    random_group_of_five = r.table.sample(5, random_state=42)
    predictions = r.get_predictions_for_group(random_group_of_five)
    agg_func = get_group_agg_func("average")
    aggregated_list = agg_func(predictions).iloc[:, :10]  # only first 10 movies

    print("\n RESULTS:")
    console = Console()
    aggregated_list_rich = Table("Aggregated List")
    aggregated_list_rich.add_row(
        aggregated_list.to_string(float_format=lambda _: "{:.4f}".format(_))
    )
    console.print(aggregated_list_rich)
    movies_df = pd.read_csv("ml-latest-small/movies.csv")
    movie_names = movies_df.set_index("movieId").loc[aggregated_list.columns]
    movie_table = Table("Movies")
    movie_table.add_row(
        movie_names.to_string(float_format=lambda _: "{:.4f}".format(_))
    )
    console.print(movie_table)


def aggregate_with_least_misery():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    random_group_of_five = r.table.sample(5, random_state=42)
    predictions = r.get_predictions_for_group(random_group_of_five)
    agg_func = get_group_agg_func("least_misery")
    aggregated_list = agg_func(predictions).iloc[:, :10]  # only first 10 movies

    print("\n RESULTS:")
    console = Console()
    aggregated_list_rich = Table("Aggregated List")
    aggregated_list_rich.add_row(
        aggregated_list.to_string(float_format=lambda _: "{:.4f}".format(_))
    )
    console.print(aggregated_list_rich)
    movies_df = pd.read_csv("ml-latest-small/movies.csv")
    movie_names = movies_df.set_index("movieId").loc[aggregated_list.columns]
    print("\n those are the most popular movies: ", movie_names)

    for mId in aggregated_list.columns:
        responsible = agg_func[mId]
        print(
            "For movie ",
            movies_df.set_index("movieId").loc[mId, "title"],
            " with rating ",
            predictions.loc[responsible[0], mId],
            " are responsible users:  ",
            responsible,
        )


def compare_aggregations():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    random_group_of_five = r.table.sample(5, random_state=42)
    predictions = r.get_predictions_for_group(random_group_of_five)
    agg_func_average = get_group_agg_func("average")
    agg_func_least_misery = get_group_agg_func("least_misery")

    aggregated_list_1 = agg_func_average(predictions).iloc[:, :10]
    aggregated_list_2 = agg_func_least_misery(predictions).iloc[:, :10]
    print(aggregated_list_1)
    print(aggregated_list_2)


def aggregate_with_custom_method():
    # This was used in the presentation
    from metric_functions import (
        get_disagreements_based_on_order,
        get_satisfaction_for_users,
        get_disagreements_for_users,
    )

    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    random_group_of_five = r.table.sample(5, random_state=42)
    predictions = r.get_predictions_for_group(random_group_of_five)
    agg_func_average = get_group_agg_func("average")
    aggregated_rankings = agg_func_average(predictions).iloc[:, :10]

    penalties, group_disagreements = get_disagreements_based_on_order(
        predictions, aggregated_rankings
    )
    satisfactions = get_satisfaction_for_users(predictions, aggregated_rankings)
    simple_disagreements = get_disagreements_for_users(satisfactions)
    print(
        "\n Disagreements (basic metric defined in the lecture)", simple_disagreements
    )

    print("Penalties: ")
    print(penalties)
    print("TOTAL GROUP DISAGREEMENTS (ours): ", group_disagreements)
    print("_" * 20)
    agg_custom = get_group_agg_func("reorder")
    new_aggregations = agg_custom(predictions, aggregated_rankings, penalties)
    penalties, group_disagreements = get_disagreements_based_on_order(
        predictions, new_aggregations
    )

    satisfactions = get_satisfaction_for_users(predictions, new_aggregations)
    simple_disagreements = get_disagreements_for_users(satisfactions)
    print("Disagreements (basic metric defined in the lecture)", simple_disagreements)
    print("Penalties: ")
    print(penalties)
    print("TOTAL GROUP DISAGREEMENTS (ours): ", group_disagreements)
    print("_" * 20)


if __name__ == "__main__":
    # test_that_CF_are_same()
    # aggregate_with_average()
    # aggregate_with_least_misery()
    aggregate_with_custom_method()
