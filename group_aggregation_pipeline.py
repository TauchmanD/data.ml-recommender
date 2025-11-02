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
    predictions = r.get_predictions_for_group_v2(single_user_table)
    prediction_movie_3 = predictions.loc[3,3]

    result = value==prediction_movie_3
    print(result)
    return result



def aggregate_with_average():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    random_group_of_five = r.table.sample(5, random_state=42)
    predictions = r.get_predictions_for_group_v2(random_group_of_five)
    agg_func = get_group_agg_func("average")
    aggregated_list = agg_func(predictions).iloc[:,: 10] # only first 10 movies

    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    print(aggregated_list)
    movie_names = movies_df.set_index('movieId').loc[aggregated_list.columns]
    print("those are the most popular movies: ", movie_names)

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
    aggregate_with_average()

