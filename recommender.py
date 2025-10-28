import pandas as pd
import numpy as np

class Recommender:
    def __init__(self, path_to_dataset: str) -> None:
        self.dataset = pd.read_csv(path_to_dataset)
        self.table = self._create_table()

    def _create_table(self):
        return self.dataset.pivot_table(index="userId", columns="movieId", values="rating")
    

    def predict(self, user_id: int, movie_id: int) -> float:
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
            if user_id == user_b:
                continue
            s = self._sim(user_id, user_b)
            results.append((user_b, s))

        if not results:
            return pd.DataFrame(columns=["neighbour", "sim"])
        
        df = pd.DataFrame(results, columns=["neighbour", "sim"])
        df = df.sort_values("sim", ascending=False)
        return df.head(n_nearest).reset_index(drop=True) if n_nearest is not None else df.reset_index(drop=True)

    

    def _sim(self, user_id_a: int, user_id_b: int) -> int:
        
        commmon = self.table.loc[user_id_a].notna() & self.table.loc[user_id_b].notna()
        matrix = self.table.columns[commmon]

        if len(matrix) == 0:
            return 0
        
        ratings_a = self.table.loc[user_id_a, matrix].astype(float)
        ratings_b = self.table.loc[user_id_b, matrix].astype(float)

        ra_mean = ratings_a.mean()
        rb_mean = ratings_b.mean()

        num = np.sum((ratings_a - ra_mean) * (ratings_b - rb_mean))
        denom = np.sqrt(np.sum((ratings_a - ra_mean)**2)) * np.sqrt(np.sum((ratings_b - rb_mean)**2))

        return 0 if denom == 0 else num / denom




