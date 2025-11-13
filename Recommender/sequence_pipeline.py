from Recommender.recommender import Recommender
from Recommender.sequence import get_sequence
from Recommender.metric_functions import get_overall_satisfaction, get_satisfaction_for_users
import pandas as pd






def test_SIAA():

    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    random_group_of_five = r.table.sample(5, random_state=24)
    preferences = r.get_predictions_for_group(random_group_of_five)

    results = {}
    satas = []

    for i in range(0,10):
        seq = get_sequence("SIAA", preferences)
        seq.k = 3 # I want only 10 movies
        seq.b = i/10

        for i in range(1,10):
            aggregated_list = next(seq)
        full_sequence = seq._history
        satisfactions = get_satisfaction_for_users(preferences,full_sequence)
        print("Satisfactions: ")
        print(satisfactions)
        satas.append(satisfactions)
        round_disagreement = satisfactions.max(axis=0) - satisfactions.min(axis=0)
        avg_disagreement = round_disagreement.mean()
        res = {
                "utility":satisfactions.values.mean(),
                "avg_disagreement": avg_disagreement
                }
        results[seq.b] = res
    
        overall_satisfactions =  get_overall_satisfaction(preferences, seq._history)
        print(overall_satisfactions)

    
    print(results)
def test_dynamic_SIAA():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    random_group_of_five = r.table.sample(5, random_state=24)
    preferences = r.get_predictions_for_group(random_group_of_five)
    seq = get_sequence("dynamicSIAA", preferences)
    seq.k = 10
    seq.siaa.k = 3
    for i in range(1,10):
        aggregated_list = next(seq)
    
    print("overall satisfactions for 10 rounds: ")
    all_rounds = seq.siaa._history
    overall_satisfactions =  get_overall_satisfaction(preferences, all_rounds)
    satisfactions = get_satisfaction_for_users(preferences,all_rounds)
    print("satisfactions", satisfactions)
    round_disagreement = satisfactions.max(axis=0) - satisfactions.min(axis=0)
    avg_disagreement = round_disagreement.mean()
    print("overall satisfactions: ", overall_satisfactions)
    print("average disagreement: ", avg_disagreement)


def main():
    # test_SIAA()
    test_dynamic_SIAA()
    print("Testing sequence generator")



if __name__ == "__main__":
    main()
