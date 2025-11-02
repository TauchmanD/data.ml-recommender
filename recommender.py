import pandas as pd
import numpy as np
from similarity_functions import get_sim

debug = False
class Recommender:
    def __init__(self, table: pd.DataFrame, sim_func="pearson") -> None:
        # self.dataset = pd.read_csv(path_to_dataset)
        self.sim_func = get_sim(sim_func)
        # self.table = self._create_table()
        self.rb_mean_cache = {}
        self.table = table

    @classmethod
    def load_from_path(cls, path_to_dataset: str, sim_func="pearson"):
        cls.dataset = pd.read_csv(path_to_dataset)
        table = cls._create_table(cls)
        return cls(table, sim_func)

    def _create_table(self):
        """Creates a table that aggregates by the ratings over the movideId columns for each user"""
        return self.dataset.pivot_table(index="userId", columns="movieId", values="rating")
    

    def predict(self, user_id: int, movie_id: int, neighbours= None, ra_mean= None) -> float:
        """Colaborative filtering implementation"""
        if ra_mean is None:
            ra_mean = float(self.table.loc[user_id].mean(skipna=True))

        if neighbours is None:
            if debug:
                print("NOT INSTANCE")
            neighbours = self.get_sim_users(user_id)
            neighbours_with_p = self.table[movie_id].dropna().index
            neighbours = neighbours[neighbours["neighbour"].isin(neighbours_with_p)].copy()

        raters = self.table[movie_id].dropna().index
        neighbours = neighbours[neighbours["neighbour"].isin(raters)]
        if neighbours.empty:
            return ra_mean

        num = 0.0
        denom = 0.0
        for _, row in neighbours.iterrows():
            if debug:
                print("predicting for movie ", movie_id)
            b = row["neighbour"]
            s = float(row["sim"])
            user_b_rating = float(self.table.loc[b, movie_id])
            # rb_mean = float(self.table.loc[b].mean(skipna=True))
            if b in self.rb_mean_cache:
                rb_mean = self.rb_mean_cache[b]
            else:
                rb_mean = float(self.table.loc[b].mean(skipna=True))
                self.rb_mean_cache[b] = rb_mean
 

            
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

    def get_predictions_for_group_v2(self, group:pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of the user×movie table where all NaNs are replaced with
        collaborative-filtering predictions (existing ratings are kept).
        """
        full_table = group.copy()

        for user_id, row_vals in full_table.iterrows():
            if debug:
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
            self.rb_mean_cache = {}

            for movie_id in missing_movies:
                full_table.loc[user_id, movie_id] = self.predict(user_id, movie_id, neighbours=neighbours, ra_mean=ra_mean)

        return full_table


