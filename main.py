from recommender import Recommender


r = Recommender("ml-latest-small/ratings.csv")
print(r.predict(3, 3))