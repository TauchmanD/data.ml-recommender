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
            return int(m.group(1))
        except ValueError:
            return None

    def _year_to_decade(year) -> str | None:
        if year is None or pd.isna(year):
            return None
        year_int = int(year)
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

        - For each user, compute the normalized distribution of their ratings
          over decades (per-user shares sum to 1).
        - Average these shares across users, so each user has equal weight.
        - Choose the decade with the highest average share.
        - Return ALL movie ids from that decade (even those not rated by the users).

        This avoids the situation where one heavy rater with thousands of
        1990s movies completely dominates a light rater.
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
        movies_df = _load_movies_df()

        # Only consider movies present in metadata
        cols = [c for c in table.columns if c in movies_df.index]
        if not cols:
            self.decade = None
            return pd.Index([], dtype=int)

        n_users = len(table.index)
        if n_users == 0:
            self.decade = None
            return pd.Index([], dtype=int)

        # Per-user normalized decade distributions
        decade_scores: dict[str, float] = {}

        for user_id, row in table.loc[:, cols].iterrows():
            rated_mask = row.notna()
            rated_movie_ids = row.index[rated_mask]
            if len(rated_movie_ids) == 0:
                # User has no ratings in the considered columns
                continue

            # Get decades for this user's rated movies
            user_movies = movies_df.loc[rated_movie_ids]
            decades = user_movies["decade"].dropna()

            if decades.empty:
                # User did not rate any movie with a known decade
                continue

            # Count how many ratings in each decade for this user
            counts = decades.value_counts()
            total = counts.sum()
            if total == 0:
                continue

            # Normalize to per-user shares (sum to 1 for that user)
            shares = counts / total

            for dec, share in shares.items():
                if dec not in self.ALLOWED_DECADES:
                    continue
                decade_scores[dec] = decade_scores.get(dec, 0.0) + float(share)

        if not decade_scores:
            self.decade = None
            return pd.Index([], dtype=int)

        # Average share per user (users with 0 in a decade implicitly contribute 0)
        avg_scores = {dec: score / n_users for dec, score in decade_scores.items()}
        decade_stats = pd.Series(avg_scores).sort_values(ascending=False)

        best_decade = decade_stats.index[0]
        self.decade = str(best_decade)

        # Return ALL movies from that decade (even those never rated by the group)
        all_decade_movies = movies_df.index[movies_df["decade"] == best_decade]
        return all_decade_movies.astype(int)


    def get_explanation_string(self, movie_id: int) -> str:
        """
        Return an explanation string ONLY if the movie is actually from self.decade.
        Otherwise return an empty string (meaning: no explanation).
        """
        movies_df = _load_movies_df()

        if movie_id not in movies_df.index or self.decade is None:
            return ""

        movie_decade = movies_df.loc[movie_id, "decade"]

        # If the movie is not from this decade, do not generate an explanation
        if pd.isna(movie_decade) or str(movie_decade) != str(self.decade):
            return ""

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
    Choose the most engaged genre for this group and return candidates
    as ALL movies of that genre.

    Engagement is computed FAIRLY per user:

        - For each user, compute a normalized distribution of their ratings
          over genres (per-user shares sum to 1).
        - Average these shares across users, so one heavy rater does not
          dominate a light rater.

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

        n_users = len(table.index)
        if n_users == 0:
            self.genre = None
            return pd.Index([], dtype=int)

        genre_scores: dict[str, float] = {}

        for user_id, row in table.loc[:, cols].iterrows():
            rated_mask = row.notna()
            rated_movie_ids = row.index[rated_mask]
            if len(rated_movie_ids) == 0:
                continue

            user_movies = movies_df.loc[rated_movie_ids]

            # Build user-level genre counts
            user_genre_counts: dict[str, int] = {}
            for mid, g_str in user_movies["genres"].items():
                if pd.isna(g_str) or g_str == "(no genres listed)":
                    continue
                genres = str(g_str).split("|")
                for g in genres:
                    user_genre_counts[g] = user_genre_counts.get(g, 0) + 1

            if not user_genre_counts:
                continue

            total = sum(user_genre_counts.values())
            if total == 0:
                continue

            # Normalize to per-user shares
            for g, cnt in user_genre_counts.items():
                share = cnt / total
                genre_scores[g] = genre_scores.get(g, 0.0) + float(share)

        if not genre_scores:
            self.genre = None
            return pd.Index([], dtype=int)

        # Average share per user
        avg_scores = {g: score / n_users for g, score in genre_scores.items()}
        genre_stats = pd.Series(avg_scores).sort_values(ascending=False)

        best_genre = genre_stats.index[0]
        self.genre = str(best_genre)

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

    r = Recommender.load_from_path("ml-latest-small/ratings.csv")
    group = r.table.sample(5, random_state=42)
    cand = Decade()
    candidates = cand(group)

    print("Best decade:", cand.decade)
    print("Number of decade candidates:", len(candidates))
    print("First 20 candidates:", candidates[:20])


def main():
    test_best_decade()


if __name__ == "__main__":
    main()
