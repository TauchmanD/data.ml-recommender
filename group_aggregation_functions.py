from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Aggregation_func(ABC):
    @abstractmethod
    def __call__(self,table:pd.DataFrame) -> pd.DataFrame:
        pass

class Average_agg(Aggregation_func):
    def __call__(self,table:pd.DataFrame) -> pd.DataFrame:
        # calculate averages for movies
        averages = table.agg(["mean"])
        order = averages.loc['mean'].sort_values(ascending=False).index

        return averages.loc[:, order]


        # raise NotImplementedError()

class Least_misery_agg(Aggregation_func):
    def __init__(self):
        self.responsibles = {}
    def __getitem__(self, key):
        return self.responsibles[key]
    def __call__(self,table:pd.DataFrame) -> pd.DataFrame:
        self.responsibles = {}
        minimums = table.agg(['min'])          # DataFrame with a single 'min' row
        mins = minimums.loc['min']             # Series view (no recompute)

        # Cache rows achieving the min (handles ties)
        self.responsibles = {
            col: table.index[table[col].eq(mins[col])].tolist()
            for col in mins.index
        }

        # Order columns by their min value (desc) and return the 1-row DataFrame
        order = mins.sort_values(ascending=False).index
        return minimums.loc[:, order]





def get_group_agg_func(func_name="average"):
    if func_name=="average":
        return Average_agg()

    if func_name == "least_misery":
        return Least_misery_agg()

if __name__=="__main__":
    agg_func = get_group_agg_func("average")
    simple_data_table = pd.DataFrame([
            [1,2,3,4,5],
            [5,4,3,2,1],
            [5,4,3,2,1],
            [5,4,3,2,1],
            [5,4,3,2,1],
            [5,4,3,2,1000]
        ], columns=("A","B","C","D","E"))
    sorted_table = agg_func(simple_data_table)
    # print(sorted_table)
    # sorted_table = sorted_table.loc[['mean']].iloc[:, :2]
    print(sorted_table)

    new_agg_func = get_group_agg_func("least_misery")
    table = new_agg_func(simple_data_table)
    print(table)
    print(new_agg_func["A"])
