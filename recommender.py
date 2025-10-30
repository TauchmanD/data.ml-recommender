import pandas as pd
import numpy as np
from similarity_functions import get_sim

class Recommender:
    def __init__(self, table: pd.DataFrame, sim_func="pearson") -> None:
        # self.dataset = pd.read_csv(path_to_dataset)
        self.sim_func = get_sim(sim_func)
        # self.table = self._create_table()
        self.table = table

    @classmethod
    def load_from_path(cls, path_to_dataset: str, sim_func="pearson"):
        cls.dataset = pd.read_csv(path_to_dataset)
        table = cls._create_table(cls)
        return cls(table, sim_func)

    def _create_table(self):
        """Creates a table that aggregates by the ratings over the movideId columns for each user"""
        return self.dataset.pivot_table(index="userId", columns="movieId", values="rating")
    

    def predict(self, user_id: int, movie_id: int) -> float:
        """Colaborative filtering implementation"""
        ra_mean = float(self.table.loc[user_id].mean(skipna=True))

        neighbours = self.get_sim_users(user_id)
        neighbours_with_p = self.table[movie_id].dropna().index
        neighbours = neighbours[neighbours["neighbour"].isin(neighbours_with_p)].copy()

        num = 0.0
        denom = 0.0
        for _, row in neighbours.iterrows():
            b = row["neighbour"]
            s = float(row["sim"])
            user_b_rating = float(self.table.loc[b, movie_id])
            rb_mean = float(self.table.loc[b].mean(skipna=True))
            num += s * (user_b_rating - rb_mean)
            denom += abs(s)
        
        pred = ra_mean if denom == 0 else ra_mean + num / denom
        return pred



    def get_sim_users(self, user_id: int, n_nearest: int = None) -> pd.DataFrame:
        results = []
        for user_b in self.table.index:
            if user_id == user_b: # dont compare user to himself
                continue
            s = self._sim(user_id, user_b)
            results.append((user_b, s))

        if not results:
            return pd.DataFrame(columns=["neighbour", "sim"])
        
        df = pd.DataFrame(results, columns=["neighbour", "sim"])
        df = df.sort_values("sim", ascending=False)
        return df.head(n_nearest).reset_index(drop=True) if n_nearest is not None else df.reset_index(drop=True)

    
    def _sim(self, user_id_a: int, user_id_b: int) -> float:
        return self.sim_func(self.table,user_id_a, user_id_b)

    def get_predictions_for_group(self, group:pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of the user×movie table where all NaNs are replaced with
        collaborative-filtering predictions (existing ratings are kept).
        """
        full_table = group.copy()

        for user_id, row_vals in full_table.iterrows():
            print("getting predictions for ", user_id)
            # Columns (movies) this user hasn't rated
            missing_movies = row_vals.index[row_vals.isna()]
            if len(missing_movies) == 0:
                continue

            # Precompute user's mean once
            ra_mean = float(self.table.loc[user_id].mean(skipna=True))

            # Precompute neighbors once (sorted by similarity)
            neighbours = self.get_sim_users(user_id)
            if neighbours.empty:
                # No neighbors -> back off to user's mean for all missing
                full_table.loc[user_id, missing_movies] = ra_mean
                continue

            # Cache neighbors' means (to avoid recomputing per movie)
            rb_mean_cache = {}

            for movie_id in missing_movies:
                # Only consider neighbors who rated this movie
                raters = self.table[movie_id].dropna().index
                neigh = neighbours[neighbours["neighbour"].isin(raters)]
                if neigh.empty:
                    full_table.loc[user_id, movie_id] = ra_mean
                    continue

                num = 0.0
                denom = 0.0
                for _, nrow in neigh.iterrows():
                    b = nrow["neighbour"]
                    s = float(nrow["sim"])

                    # neighbor's rating on this movie
                    r_bi = float(self.table.loc[b, movie_id])

                    # neighbor's mean (cached)
                    if b not in rb_mean_cache:
                        rb_mean_cache[b] = float(self.table.loc[b].mean(skipna=True))
                    rb_mean = rb_mean_cache[b]

                    num += s * (r_bi - rb_mean)
                    denom += abs(s)

                pred = ra_mean if denom == 0.0 else ra_mean + num / denom
                full_table.loc[user_id, movie_id] = pred

        return full_table


# def _sim(self, user_id_a: int, user_id_b: int) -> float:
    #     """Calculate Pearson correlacion simlilarity of two users based on their user_id"""
    #     
    #     common = self.table.loc[user_id_a].notna() & self.table.loc[user_id_b].notna() # creates boolean column matrix mask that represents which movies does user A and user B have in common
    #     matrix = self.table.columns[common] # selects those movies using the common mask
    #
    #     if len(matrix) == 0:
    #         return 0
    #     
    #     ratings_a = self.table.loc[user_id_a, matrix].astype(float) # locates ratings for user_id_a for movies in the `matrix` object
    #     ratings_b = self.table.loc[user_id_b, matrix].astype(float)
    #
    #     ra_mean = ratings_a.mean() # calculates mean value, scalar
    #     rb_mean = ratings_b.mean()
    #
    #     num = np.sum((ratings_a - ra_mean) * (ratings_b - rb_mean))
    #     denom = np.sqrt(np.sum((ratings_a - ra_mean)**2)) * np.sqrt(np.sum((ratings_b - rb_mean)**2))
    #
    #     return 0 if denom == 0 else num / denom
    #
    #
    #
    #
