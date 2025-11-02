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
    def __call__(self,table:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()




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
