from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pandas.core.frame import reorder_arrays


class Aggregation_func(ABC):
    @abstractmethod
    def __call__(self, rankings: pd.DataFrame) -> pd.DataFrame:
        pass

class Dictator_agg(Aggregation_func):

    def set_dictator(self, dictator_id):
        self.dictator_id = dictator_id

    def __call__(self, rankings: pd.DataFrame) -> pd.DataFrame:
        # rankings: users x items
        row = rankings.loc[self.dictator_id]
        order = row.sort_values(ascending=False).index
        df = row.to_frame().T  # 1 x items
        df = df.loc[:, order]
        df.index = [f"dictator_{self.dictator_id}"]
        return df

class Average_agg(Aggregation_func):
    def __call__(self, rankings: pd.DataFrame) -> pd.DataFrame:
        # calculate averages for movies
        averages = rankings.agg(["mean"])
        order = averages.loc["mean"].sort_values(ascending=False).index

        return averages.loc[:, order]

        # raise NotImplementedError()


class Least_misery_agg(Aggregation_func):
    def __init__(self):
        self.responsibles = {}

    def __getitem__(self, key):
        return self.responsibles[key]

    def __call__(self, rankings: pd.DataFrame) -> pd.DataFrame:
        self.responsibles = {}
        minimums = rankings.agg(["min"])  # DataFrame with a single 'min' row
        mins = minimums.loc["min"]  # Series view (no recompute)

        # Cache rows achieving the min (handles ties)
        self.responsibles = {
            col: rankings.index[rankings[col].eq(mins[col])].tolist()
            for col in mins.index
        }

        # Order columns by their min value (desc) and return the 1-row DataFrame
        order = mins.sort_values(ascending=False).index
        return minimums.loc[:, order]


class Remove_worst_item_agg:  # Aggregation_func):
    def __call__(
        self,
        rankings: pd.DataFrame,  # users × items predictions for the GROUP users
        aggregated_rankings: pd.DataFrame,  # 1-row DataFrame: index 'mean' or 'min', columns = current group list (top-K)
        penalties: pd.DataFrame,  # users × K penalties for the current group list (same columns as aggregated_rankings)
    ) -> pd.DataFrame:
        """
        One-step fairness-aware re-ranking:

          1) Pick the least-satisfied user (largest total penalty).
          2) Find the group item they disagree with most (max penalty).
          3) From items that user ranks higher than that offending item (and not already in the group),
             choose the candidate other users like best (according to the SAME aggregation rule).
          4) Swap it into the list and re-aggregate with the same rule; return the new 1-row DataFrame.
        """
        # --- validate / determine aggregation rule ---
        if (
            not isinstance(aggregated_rankings, pd.DataFrame)
            or aggregated_rankings.shape[0] != 1
        ):
            raise ValueError(
                "aggregated_rankings must be a 1-row DataFrame (e.g., result of .agg(['mean']) or .agg(['min'])."
            )

        agg_name = str(aggregated_rankings.index[0]).lower()
        if agg_name not in {"mean", "min"}:
            agg_name = "mean"  # default

        group_items = list(aggregated_rankings.columns)

        # Align penalties with users and group items
        penalties = penalties.reindex(index=rankings.index, columns=group_items)
        if penalties.empty or penalties.to_numpy().size == 0:
            return aggregated_rankings

        # --- 1) least-satisfied user ---
        per_user_pen = penalties.sum(axis=1, skipna=True)
        if per_user_pen.dropna().empty:
            return aggregated_rankings
        worst_user = per_user_pen.idxmax()

        # --- 2) most-disagreed item for that user ---
        per_item_pen = penalties.loc[worst_user]
        if per_item_pen.dropna().empty:
            return aggregated_rankings
        offend_item = per_item_pen.idxmax()
        print("The most disaggreed item is ", offend_item)

        # --- 3) candidates the worst user ranks higher than the offending item ---
        user_scores = rankings.loc[worst_user]
        user_ranks = user_scores.rank(ascending=False, method="first")

        if offend_item not in user_ranks or pd.isna(user_ranks[offend_item]):
            return aggregated_rankings
        r_off = user_ranks[offend_item]

        # Candidates: strictly better rank (smaller number), not already in group
        candidates = user_ranks[user_ranks < r_off].index.difference(group_items)
        if len(candidates) == 0:
            return aggregated_rankings

        # --- 4) choose candidate preferred by the rest (same aggregation rule) ---
        others = rankings.index.difference([worst_user])
        cand_scores = (
            rankings.loc[others, candidates]
            if len(others) > 0
            else pd.DataFrame(index=[], columns=candidates)
        )

        if agg_name == "mean":
            others_agg = cand_scores.mean(axis=0, skipna=True)
        else:  # 'min'
            others_agg = cand_scores.min(axis=0, skipna=True)

        # Tie-breaker by worst user's own score
        u_scores = user_scores.reindex(candidates)

        choice = pd.DataFrame({"others": others_agg, "u": u_scores}).fillna(
            float("-inf")
        )
        best_candidate = choice.sort_values(["others", "u"], ascending=False).index[0]

        # --- swap into same position and re-aggregate with same rule ---
        new_cols = group_items.copy()
        pos = new_cols.index(offend_item)
        new_cols[pos] = best_candidate

        sub = rankings.reindex(columns=new_cols)

        if agg_name == "mean":
            new_row = sub.mean(axis=0, skipna=True).to_frame().T
            new_row.index = ["mean"]
        else:  # 'min'
            new_row = sub.min(axis=0, skipna=True).to_frame().T
            new_row.index = ["min"]

        # Return sorted columns by aggregated score (desc)
        order = new_row.iloc[0].sort_values(ascending=False).index
        return new_row.loc[:, order]


def get_group_agg_func(func_name="average"):
    if func_name == "average":
        return Average_agg()

    if func_name == "least_misery":
        return Least_misery_agg()

    if func_name == "reorder":
        return Remove_worst_item_agg()

    if func_name == "dictator":
        return Dictator_agg()

if __name__ == "__main__":
    agg_func = get_group_agg_func("average")
    simple_data_table = pd.DataFrame(
        [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [5, 4, 3, 2, 1],
            [5, 4, 3, 2, 1],
            [5, 4, 3, 2, 1],
            [5, 4, 3, 2, 1000],
        ],
        columns=("A", "B", "C", "D", "E"),
    )
    sorted_table = agg_func(simple_data_table)
    # print(sorted_table)
    # sorted_table = sorted_table.loc[['mean']].iloc[:, :2]
    print(sorted_table)

    new_agg_func = get_group_agg_func("least_misery")
    table = new_agg_func(simple_data_table)
    print(table)
    print(new_agg_func["A"])
