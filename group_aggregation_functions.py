from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Aggregation_func(ABC):
    @abstractmethod
    def __call__(self,table:pd.DataFrame) -> pd.DataFrame:
        pass

class Average_agg(Aggregation_func):
    def __call__(self,table:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

class Least_misery_agg(Aggregation_func):
    def __call__(self,table:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()




def get_group_agg_func(func_name="average"):
    if func_name=="average":
        return Average_agg()

    if func_name == "least_misery":
        return Least_misery_agg()
