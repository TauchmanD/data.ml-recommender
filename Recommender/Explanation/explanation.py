from Recommender.Explanation.candidates import get_candidates
from Recommender.recommender import Recommender
import pandas as pd



def get_explanation_for_a_group(original_group_ratings: pd.DataFrame, aggregated_list: pd.Series):
    """
    This method will call all the objects from the get_candidates() method. It will create recommendations for each of them.

    Then it will look at each list of recommendation and it will compare it to the original aggregated_list.

    Then for each movie that is not in both of the lists for certain recommendation, it will provide the explanation by calling the candidates_obj.get_explanation(movie_id)

    result will be for each movie in the aggregated_list there will be a set of explanations.
    """





def main():
    pass
if __name__ == "__main__":
    main()
