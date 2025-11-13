from abc import ABC, abstractmethod
from typing import Optional
from Recommender.metric_functions import get_satisfaction_for_users, get_overall_satisfaction, get_disagreements_per_user, get_disagreements_for_users, get_disagreements_based_on_order
from Recommender.group_aggregation_functions import get_group_agg_func

import pandas as pd

class Sequence(ABC):
    def __init__(self, preferences: pd.DataFrame):
        self.preferences = preferences
        self.recommended_items = set() # cache of items that were already recommended
    @abstractmethod
    def __next__(self) -> pd.DataFrame:
        pass
    def set_top_k(self,k):
        self.k = k


class SIAA(Sequence):
    def __init__(self, preferences: pd.DataFrame, b: float = 0.5, k = 10):
        """
        Sequential Individual Adaptation Aggregation (SIAA).

        Parameters
        ----------
        preferences : DataFrame [users x movieId]
            Predicted scores for each user–item pair.
        b : float, optional (default=0.5)
            Balance parameter between overall satisfaction and disagreement
            in the weight formula:
                w = (1 - b)*(1 - satO) + b * userDis
        """
        super().__init__(preferences)
        self.b = float(b)
        self.k = k

        # First iteration uses Average aggregation (as in the paper / your skeleton)
        # self.avg_agg = get_group_agg_func("average")
        # self.avg_agg = get_group_agg_func("least_misery")
        self.avg_agg = get_group_agg_func("dictator")
        self.avg_agg.set_dictator(91)
        self.repeat = False

        # Store last round's group recommendation list Gr_{j-1}
        self.previous_recommendations: Optional[pd.DataFrame] = None

    def _available_prefs(self) -> pd.DataFrame:
        """Return preferences restricted to movies that have NOT been recommended yet."""
        if self.repeat:
            return self.preferences

        if not self.recommended_items:
            return self.preferences
        cols = [c for c in self.preferences.columns if c not in self.recommended_items]
        return self.preferences[cols]

    def __next__(self) -> pd.DataFrame:
        # ---------- FIRST ROUND: just Average aggregation on available items ----------
        if self.previous_recommendations is None:
            prefs_avail = self._available_prefs()
            first_full = self.avg_agg(prefs_avail)  # 1 x |available_items|

            # take top-k movies as the actual recommendation list Gr_1
            top_k_cols = first_full.columns[: self.k]
            first_round = first_full.loc[:, top_k_cols]

            # mark as recommended
            self.recommended_items.update(top_k_cols)

            # initialize history & previous_recommendations with ONLY the top-k
            self._history = first_round.copy()
            self.previous_recommendations = first_round.copy()

            return first_round

        # ---------- SUBSEQUENT ROUNDS j >= 2 ----------

        # 1) Overall satisfaction satO(u, GR_{j-1}) over all previous top-k lists
        overall_sat = get_overall_satisfaction(self.preferences, self._history)
        # Series indexed by userId

        # 2) Disagreement userDis(u, Gr_{j-1}) based on previous round only

        # prev_sats = get_satisfaction_for_users(self.preferences, self.previous_recommendations)
        # per_user_disagreements = get_disagreements_per_user(prev_sats)
        # My alternative, disagreements based on the order

        penalties, _ = get_disagreements_based_on_order(self.preferences, self.previous_recommendations)
        per_user_disagreements = penalties.sum(axis=1)  # Series per user
        user_dis = per_user_disagreements
        if isinstance(per_user_disagreements, pd.DataFrame):
            user_dis = per_user_disagreements.iloc[:, 0]
        else:
            user_dis = per_user_disagreements

        print("USER DISAGREEMENTS ", user_dis)
        # 3) Weights: w = (1 - b)*(1 - satO) + b * userDis
        print("SELF B is ", self.b)
        w = (1.0 - self.b) * (1.0 - overall_sat) + self.b * user_dis
        w = w.reindex(self.preferences.index).fillna(0.0)

        # 4) Compute SIAA scores *only on not-yet-recommended items*
        prefs_avail = self._available_prefs()
        weighted_prefs = prefs_avail.mul(w, axis=0)
        group_scores = weighted_prefs.sum(axis=0).sort_values(ascending=False)

        if group_scores.empty:
            # no more items left to recommend
            raise StopIteration

        # build full ranking for available items
        full_new_round = group_scores.to_frame().T

        # take top-k from the *remaining* items
        top_k_cols = full_new_round.columns[: self.k]
        new_round = full_new_round.loc[:, top_k_cols]

        # mark these as recommended
        self.recommended_items.update(top_k_cols)

        # give the round a name
        round_idx = self._history.shape[0] + 1
        new_round.index = [f"siaa_{round_idx}"]

        # update history GR_j = GR_{j-1} ∪ {Gr_j}  (only with the top-k columns)
        self._history = pd.concat([self._history, new_round], axis=0, join="outer")

        # update last round for disagreement calculation
        self.previous_recommendations = new_round.copy()

        return new_round


class dynamicSIAA(Sequence):
    def __init__(
        self,
        preferences: pd.DataFrame,
        b_low: float = 0.3,
        b_high: float = 0.6,
        disagreement_threshold: float = 0.10,
        k: int = 3,
    ):
        # We still call Sequence.__init__ so the type is consistent,
        # but all logic is delegated to the internal SIAA instance.
        super().__init__(preferences)
        self.repeat = False

        # base SIAA with some default b and k (b will be overwritten dynamically)
        self.siaa = SIAA(preferences, b=b_low, k=k)
        self.siaa.repeat = self.repeat

        # dynamic-b params
        self.b_low = float(b_low)
        self.b_high = float(b_high)
        self.disagreement_threshold = float(disagreement_threshold)

        # remember the last recommendation round (1×K DataFrame)
        self._last_round: pd.DataFrame | None = None

    def set_repeat(self, bol:bool):
        self.siaa.repeat = bol

    def __next__(self) -> pd.DataFrame:
        """
        On each round j:
          - if j > 1, compute disagreement of previous round
            and set siaa.b accordingly (b_low / b_high).
          - then delegate to SIAA to produce the next group ranking.
        """
        # For the first round: just use whatever b SIAA was initialized with.
        if self._last_round is not None:
            # 1) satisfaction in the previous round for all users
            prev_sats = get_satisfaction_for_users(self.preferences, self._last_round)
            # 2) scalar group disagreement for that round
            group_disagreement = get_disagreements_for_users(prev_sats)

            # penalties, _ = get_disagreements_based_on_order(self.preferences, self.previous_recommendations)
            # per_user_disagreements = penalties.sum(axis=1)  # Series per user
            # user_dis = per_user_disagreements
            # 3) adapt b based on disagreement
            print("GROUP DISAGREEMENT IS ", group_disagreement)
            if group_disagreement > self.disagreement_threshold:
                # unfair round → react strongly to disagreement
                self.siaa.b = self.b_high
                print("HIGH disagreement -> raising B")
            else:
                # fairly balanced → favor stability / long-term satisfaction
                print("low disagreement -> lowering B")
                self.siaa.b = self.b_low

        # 4) ask SIAA for the next recommendation round
        new_round = next(self.siaa)   # 1×K DataFrame, row index = e.g. 'siaa_2', cols = movieIds

        # 5) remember it for the next iteration
        self._last_round = new_round

        return new_round

    def set_top_k(self, k: int):
        self.siaa.k = int(k)


def get_sequence(sequence_name, preferences):
    if sequence_name == "custom":
        return CustomSequence(preferences)
    elif sequence_name == "SIAA":
        return SIAA(preferences)
    elif sequence_name == "dynamicSIAA":
        return dynamicSIAA(preferences)
    else:
        raise NotImplementedError()



def test_siaa():
    preferences = pd.DataFrame(
        [
            [1, 1, 0.8],
            [1, 2, 0.1],
            [1, 3, 0.4],
            [1, 4, 0.9],
            [2, 1, 0.3],
            [2, 2, 0.9],
            [2, 3, 0.1],
            [2, 4, 0.1],
            [3, 1, 0.8],
            [3, 2, 0.4],
            [3, 3, 0.4],
            [3, 4, 0.1],
            [4, 1, 0.3],
            [4, 2, 0.5],
            [4, 3, 1.0],
            [4, 4, 0.1],
        ],
        columns=["userId", "movieId", "rating"],
    )
    preferences = preferences.pivot_table(
        index="userId", columns="movieId", values="rating"
    )
    seq = get_sequence("SIAA", preferences)
    seq.k = 2
    aggregations = next(seq)
    print("aggregations")
    print(aggregations)
    aggregations = next(seq)
    print("aggregations")
    print(aggregations)


def test_custom_sequence():
    preferences = pd.DataFrame(
        [
            [1, 1, 0.8],
            [1, 2, 0.1],
            [1, 3, 0.4],
            [1, 4, 0.9],
            [2, 1, 0.3],
            [2, 2, 0.9],
            [2, 3, 0.1],
            [2, 4, 0.1],
            [3, 1, 0.8],
            [3, 2, 0.4],
            [3, 3, 0.4],
            [3, 4, 0.1],
            [4, 1, 0.3],
            [4, 2, 0.5],
            [4, 3, 1.0],
            [4, 4, 0.1],
        ],
        columns=["userId", "movieId", "rating"],
    )
    preferences = preferences.pivot_table(
        index="userId", columns="movieId", values="rating"
    )

    seq = get_sequence("custom", preferences)
    print(next(seq))

if __name__ == "__main__":
    test_siaa()
