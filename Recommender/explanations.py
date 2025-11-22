from Recommender.recommender import Recommender
from rich import print
import pandas as pd


def get_item_intensities(group: pd.DataFrame):
    "For each item returns how many users rated them"

    rating_count = group.notnull().sum(axis = 0)
    return rating_count 

def get_user_intensities(group: pd.DataFrame):
    intensities = group.notnull().sum(axis = 1)
    return intensities

def compute_popularity_distribution(item_intensities: pd.DataFrame):
    distribution = item_intensities.value_counts()
    return distribution


def main():
    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    random_group_of_five = r.table.sample(5, random_state=42)
    item_intensities = get_item_intensities(random_group_of_five)
    popularity_distribution = compute_popularity_distribution(item_intensities)

    user_intensities = get_user_intensities(random_group_of_five)

    print("Item intensity: ")
    print(item_intensities)

    print("Item intensity distribution: ")
    print(popularity_distribution)

    print("User intensity: ")
    print(user_intensities)
    

if __name__ == "__main__":
    main()
