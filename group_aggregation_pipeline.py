from recommender import Recommender
import time
from group_aggregation_functions import get_group_agg_func




def main():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    group = Recommender(r.table.sort_index().head(10).copy())
    aggregation_function = get_group_agg_func("average")
    print("starting calculation")
    time_start = time.time()
    group_predictions = group.get_predictions_for_group(r.table.sort_index().head(10).copy())
    time_end = time.time()
    ranked_predicitons = aggregation_function(group_predictions)
    print("time for predictions: ", time_end-time_start)




if __name__ == "__main__":
    main()
