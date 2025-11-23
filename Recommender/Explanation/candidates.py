from abc import ABC, abstractmethod
import pandas as pd 

def get_movie_name_from_id(movie_ids: pd.Series):
    """returns a dataframe consisted of movie names for the movie ids"""
    movies_df = pd.read_csv("ml-latest-small/movies.csv")
    movie_names = movies_df.set_index("movieId").loc[movie_ids]

    return movie_names

class Candidates(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(table: pd.DataFrame) -> pd.DataFrame:
        """Returns pandas serius consisted of item ids"""
        pass
    @abstractmethod
    def get_explanation_string(movie_id):
        """Returns string that gives user the explanation for the certain movie"""
        pass



class Decade(Candidates):
    """This generates candidates based on the decade of the movie, for each decade it will """
    def __init__(self):
        super().__init__()
        self.decade = None

    
    def __call__(self, table: pd.Dataframe):
        """
        For each decade it will compute the average group popularity 
        and it will choose the highest decade. It will return all the movies ids 
        from that decace (even those ones, that were not rated by the users).

        It will rate 50s, 60s, 70s, 80s, 90s, 2000s, 2010s, 2020s
        """
        raise NotImplementedError
    def get_explanation_string(self,movie_id):
        movie_name = get_movie_name_from_id(movie_id)
        return f"If group did not rate the movies from the {self.decade}, then movie {movie_name} would not be recommended"

class Most_agreed_items(Candidates):
    def __init__(self):
        super().__init__()
        self.best_variance = None
        self.agreed_movies = None

    def __call__(self, table: pd.DataFrame):
        """
        Top k = 5
        This class will search through the table and it will identify:
            1. which movies are most rated
            2. What is the variance in the rating

        The candidates will be chosen by sorting first by popularity (how many ratings from the group the movie has) and than by variance in rating. Top 5 movies will be returned

        """
        raise NotImplementedError

    def get_explanation_string(self, movie_id):
        aggreed_names = get_movie_name_from_id(self.agreed_movies)
        explanation_movie = get_movie_name_from_id(movie_id)
        return f"If your group did not agree so much on the rating of {aggreed_names}, than movie {explanation_movie} would not be recommended"

class GenresNotRated(Candidates):
    def __init__(self):
        super().__init__()
        self.genre = None
    def __call__(self, table: pd.DataFrame):
        """Choose the most often rated genre and return candidates (all movies) that are that genre"""
        raise NotImplementedError
    def get_explanation_string(self, movie_id): 
        movie_name = get_movie_name_from_id(movie_id)
        return f"If your group did not watch any {self.genre} movies, than movie {movie_name} would not be recommended"
        

def get_candidates():
    return [Decade(), Most_agreed_items(), GenresNotRated]

def main():
    pass
if __name__ == "__main__":
    main()









