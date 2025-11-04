from recommender import Recommender
import pandas as pd
import random
import time
from group_aggregation_functions import get_group_agg_func


def test_that_CF_are_same():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    value = r.predict(3, 3)

    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    single_user_table = r.table.loc[[3]]
    predictions = r.get_predictions_for_group(single_user_table)
    prediction_movie_3 = predictions.loc[3,3]

    result = value==prediction_movie_3
    print(result)
    return result



def aggregate_with_average():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    random_group_of_five = r.table.sample(5, random_state=42)
    predictions = r.get_predictions_for_group(random_group_of_five)
    agg_func = get_group_agg_func("average")
    aggregated_list = agg_func(predictions).iloc[:,: 10] # only first 10 movies

    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    print(aggregated_list)
    movie_names = movies_df.set_index('movieId').loc[aggregated_list.columns]
    print("those are the most popular movies: ", movie_names)

def aggregate_with_least_misery():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    random_group_of_five = r.table.sample(5, random_state=42)
    predictions = r.get_predictions_for_group(random_group_of_five)
    agg_func = get_group_agg_func("least_misery")
    aggregated_list = agg_func(predictions).iloc[:,: 10] # only first 10 movies

    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    print(aggregated_list)
    movie_names = movies_df.set_index('movieId').loc[aggregated_list.columns]
    print("those are the most popular movies: ", movie_names)

    for mId in aggregated_list.columns:
        responsible = agg_func[mId]
        print("For movie ", movies_df.set_index('movieId').loc[mId, "title"], " with rating ", predictions.loc[responsible[0],mId], " are responsible users:  ", responsible)
    

def compare_aggregations():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    random_group_of_five = r.table.sample(5, random_state=42)
    predictions = r.get_predictions_for_group(random_group_of_five)
    agg_func_average = get_group_agg_func("average")
    agg_func_least_misery = get_group_agg_func("least_misery")

    aggregated_list_1 = agg_func_average(predictions).iloc[:,: 10]
    aggregated_list_2 = agg_func_least_misery(predictions).iloc[:,: 10]
    print(aggregated_list_1)
    print(aggregated_list_2)


def main():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    group = Recommender(r.table.sort_index().head(10).copy())
    aggregation_function = get_group_agg_func("average")
    time_start = time.time()
    group_predictions = group.get_predictions_for_group(r.table.sort_index().head(10).copy())
    time_end = time.time()
    ranked_predicitons = aggregation_function(group_predictions)
    print("time for predictions: ", time_end-time_start)




if __name__ == "__main__":
    # main()
    aggregate_with_least_misery()
