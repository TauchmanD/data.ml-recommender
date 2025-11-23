from abc import ABC, abstractmethod
import re
import pandas as pd


_MOVIES_DF: pd.DataFrame | None = None


def _load_movies_df() -> pd.DataFrame:
    """Load movies.csv once, parse year and decade, and cache it."""
    global _MOVIES_DF
    if _MOVIES_DF is not None:
        return _MOVIES_DF

    movies_df = pd.read_csv("ml-latest-small/movies.csv")

    def _extract_year(title: str) -> int | None:
        m = re.search(r"\((\d{4})\)", str(title))
        if not m:
            return None
        try:
            # Make sure it's an INT, not float
            return int(m.group(1))
        except ValueError:
            return None

    def _year_to_decade(year) -> str | None:
        if year is None or pd.isna(year):
            return None
        year_int = int(year)  # <--- critical: cast to int to avoid 1990.0
        if year_int < 1950 or year_int > 2029:
            return None
        decade_start = (year_int // 10) * 10
        return f"{decade_start}s"  # e.g. "1990s"

    movies_df["year"] = movies_df["title"].apply(_extract_year)
    movies_df["decade"] = movies_df["year"].apply(_year_to_decade)
    movies_df = movies_df.set_index("movieId")

    _MOVIES_DF = movies_df
    return _MOVIES_DF


def get_movie_name_from_id(movie_ids):
    """
    Returns movie title(s) for the given movie id(s).

    - If movie_ids is scalar, returns a single string.
    - If movie_ids is list/Series/Index, returns a Series of titles.
    """
    movies_df = _load_movies_df()

    if isinstance(movie_ids, (list, tuple, pd.Index, pd.Series)):
        return movies_df.loc[movie_ids, "title"]

    # scalar
    movie_id_int = int(movie_ids)
    return str(movies_df.loc[movie_id_int, "title"])


class Candidates(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, table: pd.DataFrame) -> pd.Index:
        """Return a pandas Index (or list-like) of candidate item ids."""
        pass

    @abstractmethod
    def get_explanation_string(self, movie_id: int) -> str:
        """Return a human-readable explanation string for the given movie."""
        pass


class Decade(Candidates):
    """
    This generates candidates based on the decade of the movie.

    __call__:
        For the given group rating table it will:
        - For each decade, compute the average group popularity
          (average #ratings per movie in that decade, restricted to movies the group touched).
        - Choose the decade with the highest average popularity.
        - Return ALL movie ids from that decade (even those not rated by the users).

        It considers: 50s, 60s, 70s, 80s, 90s, 2000s, 2010s, 2020s.
    """

    ALLOWED_DECADES = [
        "1950s",
        "1960s",
        "1970s",
        "1980s",
        "1990s",
        "2000s",
        "2010s",
        "2020s",
    ]

    def __init__(self):
        super().__init__()
        self.decade: str | None = None

    def __call__(self, table: pd.DataFrame) -> pd.Index:
        print("searching for candidates based on decade")
        movies_df = _load_movies_df()

        # Only consider movies that are in both the group's table and metadata
        cols = [c for c in table.columns if c in movies_df.index]
        if not cols:
            print("not cols")
            return pd.Index([], dtype=int)

        # Group popularity for each movie (how many users rated it)
        intensities = table.loc[:, cols].notnull().sum(axis=0)
        print("intensities: ", intensities)

        # Attach decade info
        sub = movies_df.loc[cols].copy()
        sub["intensity"] = intensities
        sub = sub[sub["decade"].notna()]
        print("sub: ", sub.head())

        if sub.empty:
            print("sub empty")
            return pd.Index([], dtype=int)

        # Average intensity per decade (use whatever decades actually appear)
        decade_stats = sub.groupby("decade")["intensity"].mean()
        decade_stats = decade_stats.dropna()

        if decade_stats.empty:
            print("decade stats empty: ", decade_stats)
            return pd.Index([], dtype=int)

        best_decade = decade_stats.idxmax()
        self.decade = str(best_decade)
        print("The best decade is: ", self.decade)

        # Return ALL movies from that decade (even those never rated by the group)
        all_decade_movies = movies_df.index[movies_df["decade"] == best_decade]
        return all_decade_movies.astype(int)

    def get_explanation_string(self, movie_id: int) -> str:
        movie_name = get_movie_name_from_id(movie_id)
        return (
            f"If your group had not rated movies from the {self.decade}, "
            f"then movie '{movie_name}' would not be recommended."
        )


class Most_agreed_items(Candidates):
    """
    This candidate generator picks the movies where the group both:
    - rated them the most, and
    - agreed the most (low rating variance).

    It returns the top-k=5 movies sorted by:
        1) popularity (intensity: #group ratings) descending
        2) variance ascending (lower variance = more agreement)
    """

    def __init__(self):
        super().__init__()
        self.best_variance: float | None = None
        self.agreed_movies: list[int] | None = None
        self.k: int = 5

    def __call__(self, table: pd.DataFrame) -> pd.Index:
        # How many users rated each movie
        intensities = table.notnull().sum(axis=0)

        # Require at least 2 ratings to talk about "agreement"
        mask = intensities >= 2
        cols = mask.index[mask]
        if len(cols) == 0:
            self.agreed_movies = []
            self.best_variance = None
            return pd.Index([], dtype=int)

        sub = table.loc[:, cols]

        # Variance per movie (across users), ignore NaN, population variance
        variances = sub.var(axis=0, ddof=0)

        df = pd.DataFrame(
            {
                "intensity": intensities[cols],
                "variance": variances,
            }
        )

        # Most rated first, then lowest variance
        df = df.sort_values(by=["intensity", "variance"], ascending=[False, True])

        top = df.head(self.k)
        self.agreed_movies = [int(m) for m in top.index]
        self.best_variance = float(top["variance"].min()) if not top.empty else None

        return top.index.astype(int)

    def get_explanation_string(self, movie_id: int) -> str:
        explanation_movie = get_movie_name_from_id(movie_id)

        if not self.agreed_movies:
            agreed_part = "your strongest shared favourites"
        else:
            names = get_movie_name_from_id(self.agreed_movies)
            if isinstance(names, pd.Series):
                # Show at most 3 titles in explanation
                subset = names.tolist()[:3]
                agreed_part = ", ".join(f"'{n}'" for n in subset)
                if len(names) > 3:
                    agreed_part += ", and others"
            else:
                agreed_part = f"'{names}'"

        return (
            f"If your group did not agree so strongly on movies like {agreed_part}, "
            f"then movie '{explanation_movie}' would not be recommended."
        )


class GenresNotRated(Candidates):
    """
    Choose the most often-rated genre for this group and return candidates
    as ALL movies of that genre.

    This corresponds to:
        "If your group had 0 engagement with this genre, then movie B would
        not be recommended."
    """

    def __init__(self):
        super().__init__()
        self.genre: str | None = None

    def __call__(self, table: pd.DataFrame) -> pd.Index:
        movies_df = _load_movies_df()

        # Only consider movies that appear in this group's columns
        cols = [c for c in table.columns if c in movies_df.index]
        if not cols:
            self.genre = None
            return pd.Index([], dtype=int)

        intensities = table.loc[:, cols].notnull().sum(axis=0)

        # We will sum intensity per genre
        genre_popularity: dict[str, int] = {}

        for movie_id in cols:
            genres_str = movies_df.loc[movie_id, "genres"]
            if pd.isna(genres_str) or genres_str == "(no genres listed)":
                continue
            genres = str(genres_str).split("|")
            weight = int(intensities[movie_id])
            for g in genres:
                genre_popularity[g] = genre_popularity.get(g, 0) + weight

        if not genre_popularity:
            self.genre = None
            return pd.Index([], dtype=int)

        # Most engaged genre
        best_genre = max(genre_popularity.items(), key=lambda kv: kv[1])[0]
        self.genre = best_genre

        # Candidates: ALL movies with that genre (not just those the group rated)
        def has_genre(gs: str) -> bool:
            return best_genre in str(gs).split("|")

        mask = movies_df["genres"].apply(has_genre)
        candidate_ids = movies_df.index[mask]

        return candidate_ids.astype(int)

    def get_explanation_string(self, movie_id: int) -> str:
        movie_name = get_movie_name_from_id(movie_id)
        return (
            f"If your group had not watched any {self.genre} movies, "
            f"then movie '{movie_name}' would not be recommended."
        )


def get_candidates() -> list[Candidates]:
    return [Decade(), Most_agreed_items(), GenresNotRated()]

def test_best_decade():
    from Recommender.recommender import Recommender
    from Recommender.group_aggregation_functions import get_group_agg_func

    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    group = r.table.sample(5, random_state=42)
    cand = Decade()
    candidates = cand(group)

    print(candidates)


    

def main():
    test_best_decade()


if __name__ == "__main__":
    main()
