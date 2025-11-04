from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Sim_func(ABC):
    @abstractmethod
    def __call__(self, table: pd.DataFrame, user_id_a: int, user_id_b: int) -> float:
        pass


class Pearson_correlation(Sim_func):
    def __call__(self, table: pd.DataFrame, user_id_a: int, user_id_b: int) -> float:
        common = (
            table.loc[user_id_a].notna() & table.loc[user_id_b].notna()
        )  # creates boolean column matrix mask that represents which movies does user A and user B have in common
        matrix = table.columns[common]  # selects those movies using the common mask

        if len(matrix) == 0:
            return 0

        ratings_a = table.loc[user_id_a, matrix].astype(
            float
        )  # locates ratings for user_id_a for movies in the `matrix` object
        ratings_b = table.loc[user_id_b, matrix].astype(float)

        ra_mean = ratings_a.mean()  # calculates mean value, scalar
        rb_mean = ratings_b.mean()

        num = np.sum((ratings_a - ra_mean) * (ratings_b - rb_mean))
        denom = np.sqrt(np.sum((ratings_a - ra_mean) ** 2)) * np.sqrt(
            np.sum((ratings_b - rb_mean) ** 2)
        )

        return 0 if denom == 0 else num / denom


class PD_SignificanceWeightedPearson(Sim_func):
    """
    Popularity-Discounted, Significance-Weighted Pearson similarity.

    - Popularity discount (item weights): penalizes very popular items so niche overlap matters more.
      weight modes:
        * "log"  : w_i = log(1 + N_users / n_i)          (default)
        * "sqrt" : w_i = 1 / sqrt(n_i)
        * None   : w_i = 1 (standard Pearson)
    - Significance weighting: shrinks similarities from small co-rated sets.
      s(u,v) = (m / (m + alpha)) * rho_w, where m = |co-rated items|.

    Parameters
    ----------
    alpha : float
        Strength of significance shrinkage. Typical 5–20.
    min_common : int
        Minimum number of co-rated items required to compute a nonzero similarity.
    weight_mode : {"log","sqrt",None}
        Item popularity weighting scheme.
    eps : float
        Numerical stability constant.
    """

    def __init__(
        self,
        alpha: float = 5.0,
        min_common: int = 3,
        weight_mode: str | None = "log",
        eps: float = 1e-12,
    ):
        self.alpha = float(alpha)
        self.min_common = int(min_common)
        self.weight_mode = weight_mode
        self.eps = float(eps)

    def __call__(self, table: pd.DataFrame, user_id_a: int, user_id_b: int) -> float:
        # Columns both users rated
        common = table.loc[user_id_a].notna() & table.loc[user_id_b].notna()
        cols = table.columns[common]
        m = len(cols)
        if m < self.min_common:
            return 0.0

        # Ratings for co-rated items
        ra = table.loc[user_id_a, cols].astype(float)
        rb = table.loc[user_id_b, cols].astype(float)

        # User means over ALL items each user rated (more stable than co-rated mean)
        mu_a = table.loc[user_id_a].astype(float).mean(skipna=True)
        mu_b = table.loc[user_id_b].astype(float).mean(skipna=True)

        xa = ra - mu_a
        xb = rb - mu_b

        # Item popularity counts across ALL users for the selected items
        item_count = table[cols].notna().sum(axis=0).astype(float)
        N_users = float(table.shape[0])

        # Weights (popularity discount)
        if self.weight_mode == "log":
            w = np.log1p(N_users / (item_count + self.eps))
        elif self.weight_mode == "sqrt":
            w = 1.0 / np.sqrt(item_count + self.eps)
        else:
            w = np.ones_like(item_count.values, dtype=float)

        xa_vals = xa.to_numpy()
        xb_vals = xb.to_numpy()
        w_vals = np.asarray(w, dtype=float)

        num = np.sum(w_vals * xa_vals * xb_vals)
        den = np.sqrt(np.sum(w_vals * xa_vals * xa_vals)) * np.sqrt(
            np.sum(w_vals * xb_vals * xb_vals)
        )

        if den <= self.eps:
            return 0.0

        rho_w = num / den

        # Significance weighting
        sig = m / (m + self.alpha)
        s = sig * rho_w

        # Clip to valid correlation range
        return float(np.clip(s, -1.0, 1.0))


def get_sim(sim_name="pearson"):
    if sim_name == "pearson":
        return Pearson_correlation()
    if sim_name == "PD_SignificanceWeightedPearson":
        return PD_SignificanceWeightedPearson()
