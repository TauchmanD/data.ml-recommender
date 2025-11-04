from recommender import Recommender


r = Recommender.load_from_path("ml-latest-small/ratings.csv")
print(r.predict(3, 3))
