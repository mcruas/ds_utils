import functools
from typing import Iterable
import polars as pl

def join_all_dataframes(lst:Iterable, index) -> pl.DataFrame:
    """Joins all dataframes in a list, using the 'index' column as key"""
    return functools.reduce(lambda x, y: x.join(y, how='outer', on=index), 
                 [df for df in lst])


def summary_statistics(df, category:str=None) -> pl.DataFrame:
    """Gets all columns statistics for a class of data (numeric or categorical).

    Args:
        df (pl.DataFrame): Polars DataFrame to summarize.
        category (str): 'numeric', 'categorical', or 'date'

    Returns:
        pl.DataFrame: Summary of statistics.
    """

    def stats_categorical(s) -> dict:
        s_freq = s.filter(s.is_not_null()).value_counts().sort(by='counts')
        return {
            'Count Total': s.len(),
            'Count NULL': s.is_null().sum(),
            '% NULL': f"{s.is_null().sum() / s.len():.2%}",
            'Distinct values': s.n_unique(),
            'Most frequent value': f"{s_freq[-1,0]} ({s_freq[-1,1]})",
            'Least frequent value': f"{s_freq[0,0]} ({s_freq[0,1]})",
        }

    def apply_stats_func(df, func) -> pl.DataFrame:
        registers = {col_name: func(df[col_name]) for col_name in df.columns}
        return pl.DataFrame(
            # change format of dict so it can be interpreted by polars
            [{'variable': k, **v} for k, v in registers.items()]
        ).sort(by='variable')

    if category is None:
        summary_statistics(df, 'numeric')
        summary_statistics(df, 'categorical')
        summary_statistics(df, 'date')

    cols_numeric = [col for col in df.columns if df[col].dtype in (pl.datatypes.Float64, pl.datatypes.Int64)]
    cols_date = [col for col in df.columns if df[col].dtype in (pl.datatypes.Date, pl.datatypes.Datetime)]
    cols_categorical = list(set(df.columns) - set(cols_date + cols_numeric))
    
    if category == 'categorical':
        return apply_stats_func(df.select(cols_categorical), stats_categorical)
    elif category == 'numeric':
        return df.select(cols_numeric).describe()
    elif category == 'date':
        return df.select(cols_date).describe()
        # return apply_stats_func(df.select(cols_date), stats_date)
    elif category == 'all':
        return {
            'categorical': summary_statistics(df.select(cols_categorical), 'categorical'),
            'numeric': summary_statistics(df.select(cols_numeric), 'numeric'),
            'date': summary_statistics(df.select(cols_date), 'date'),
        }