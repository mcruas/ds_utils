import pandas as pd
import numpy as np
import string
from misc import list_difference, list_intersection


def create_df_eda(df, TIME_VARIABLES, cols_eda=None, lags=False, cuts:dict=False):

    def create_cuts(df, cuts, cols_eda, lags=None):
    # creates cuts for continuous variables which are on the cuts dictionary
        for col, n_cuts, round_to in cuts:
            if col not in cols_eda:
                continue
            # df[col] = pd.qcut(df[col], n_cuts, duplicates='drop')
            # print(col)
            df[col] = pretty_qcut(df[col], n_cuts, round_to=round_to, break_at_zero=True)
            # print('done!')
            if lags:
                for lag in lags:
                    s = pretty_qcut(df[f'{col}_L{lag}'], n_cuts, round_to=round_to, break_at_zero=True)
                    df[f'{col}_L{lag}'] = s
        return df

    if cols_eda is None:
        cols_eda = list_difference(df.columns, ['cpf_cnpj', 'month'])

    cols_make_lags = list_intersection(cols_eda, TIME_VARIABLES)
    df = df[cols_eda + ['cpf_cnpj', 'month']]

    if lags:
        df = create_lags(df, time='month', lags=lags, id='cpf_cnpj', vars=cols_make_lags)

    if cuts:
        df = create_cuts(df, cuts, cols_eda, lags)

    return df


def create_lags(df, time : str, lags = 1, id : str ='cpf_cnpj',  vars=None,
    drop_max_lags=True, fillna:bool=True) -> pd.DataFrame: 
# Creates lags of a variables given in vars. Returns the same dataframe with the new variables.
# time is the column with the time variable (for now, only month is supported)
# lags is the list of lags to create. A single 
# If drop_max_lags is True, dates which are before 
#   add_month(min(date)+ max(lags)) are dropped .
# If fillna is True, the lags will be filled with the 0 values.

    def add_month(date, n):
        return (pd.to_datetime(date) + pd.DateOffset(months=n))

    def drop_lags(df, time, max_lag):
    # drops rows which correspond to the last lags
        min_date = add_month(df[time].min(), max_lag)
        return df.query(f"{time} > @min_date")

    def create_single_lag(df, time:str, lag:int = 1, id:str='cpf_cnpj',  
    vars:list=None, fillna:bool=True) -> pd.DataFrame:
    # function that creates a single lag. Returns the same dataframe with the new variables.
    # variables description are the same as parent function
        if vars is None:
            vars = df.columns.drop(time).drop(id)
        df_merge = df[[id, time] + vars].copy()
        df_merge[time] = add_month(df_merge[time], lag)
        to_new_var_names_dict = {var: var+'_L'+str(lag) for var in vars}
        df_merge = df_merge.rename(columns=to_new_var_names_dict)
        df = df.merge(df_merge, on=[time,id], how='left')
        if fillna:
            new_var_names = to_new_var_names_dict.values()
            df.loc[:,new_var_names] = df[new_var_names].fillna(0)
        return df

    df[time] = pd.to_datetime(df[time])

    if isinstance(lags,int):
        df = create_single_lag(df, time, lags, id, vars, fillna)
        if drop_max_lags:
            df = drop_lags(df, time, lags)
        return df
    elif isinstance(lags,list):
        for lag in lags:
            df = create_single_lag(df, time, lag, id, vars, fillna)
        if drop_max_lags:
            df = drop_lags(df, time, max(lags))
        return df
    else:
        raise ValueError('lags must be int or list')


def generate_mock_dataframe(n_rows:int=100, n_columns:int=6):
    """Generates a mock dataframe with a specified number of rows and columns.

    Args:
        n_rows (int, optional): Number of rows. Defaults to 100.
        n_columns (int, optional): Number of columns. Defaults to 6.
    """

    def round_up_to_nearest_multiple_of_4(number):
        if number % 4 == 0:
            return number
        else:
            return number + (4 - (number % 4))

    n_columns = round_up_to_nearest_multiple_of_4(n_columns)

    num_each_col_type = n_columns // 4

    # Generate numeric (float) columns
    numeric_data = np.random.rand(n_rows, num_each_col_type)

    # Generate integer columns
    integer_data = np.random.randint(1, 100, size=(n_rows, num_each_col_type))

    # Generate categorical columns
    categorical_data = np.empty((n_rows, num_each_col_type), dtype=object)
    for i in range(num_each_col_type):
        categorical_data[:, i] = np.random.choice(["A", "B", "C", "D"], n_rows)

    # Generate datetime columns
        start_date = pd.Timestamp("2000-01-01")
        end_date = pd.Timestamp("2022-01-01")
        date_range = (end_date - start_date).days
        datetime_data = np.array([start_date + pd.Timedelta(days=np.random.randint(0, date_range))
                                for _ in range(n_rows * num_each_col_type)]).reshape(n_rows, num_each_col_type)

    # Combine numeric, integer, and categorical columns
    data = np.column_stack((numeric_data, integer_data, categorical_data, datetime_data))

    # Generate column names
    column_names = (
        [f"num_{i}" for i in range(num_each_col_type)] +
        [f"int_{i}" for i in range(num_each_col_type)] +
        [f"cat_{i}" for i in range(num_each_col_type)] + 
        [f"date_{i}" for i in range(num_each_col_type)]
    )

    # Create a Pandas dataframe
    df = pd.DataFrame(data, columns=column_names)

    return df


def pretty_qcut(x:pd.Series, bins:int=5, round_to:int=0, break_at_zero:bool=False, offset_letter:int=0, plot:bool=False, **kwargs):
# returns a pretty quantile cut, with labels roundings to the nearest integer or float
    def def_bin(i, bins, round_to):
        if i == 0 and round_to <= 0:
            return f"{string.ascii_uppercase[offset_letter+i]}. [{bins[i]:.0f},{bins[i+1]:.0f}]"
        elif i == 0 and round_to > 0:
            return f"{string.ascii_uppercase[offset_letter+i]}. [{bins[i]:.{round_to}f},{bins[i+1]:.{round_to}f}]"
        elif i > 0 and round_to <= 0:
            return f"{string.ascii_uppercase[offset_letter+i]}. ({bins[i]:.0f},{bins[i+1]:.0f}]"
        else:
            return f"{string.ascii_uppercase[offset_letter+i]}. ({bins[i]:.{round_to}f},{bins[i+1]:.{round_to}f}]"

    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    if isinstance(bins, int):
        bins = x.quantile(np.linspace(0, 1, bins + 1)).round(round_to).drop_duplicates()
        if break_at_zero and (bins[0] < 0) and (bins.values[-1] > 0):
            bins = bins.append(pd.Series(0), ignore_index=True).sort_values().drop_duplicates()
        # bins[0] = x.min()
        # bins[-1] = x.max()
        bins = bins.to_list()

    labels = [def_bin(i, bins, round_to) for i,j in enumerate(bins[:-1])]

    return pd.cut(x, bins=bins, labels=labels, duplicates='drop', include_lowest=True)


def qcut_except_zero(x, bins_less_0, bins_more_0, round_to:int=0):
# Cuts a variable into bins, except for the bins with 0. Uses pretty_qcut.
# bins_less_0 and bins_more_0 are the number of bins for the bins with values less than 0 and greater than 0, respectively.
# round_to is the number of decimal places to round the bins to.
    output = x.copy()
    output[x < 0] = pretty_qcut(x[x < 0], bins=bins_less_0, round_to=round_to)
    n_less_0 = output[x < 0].nunique()
    output[x == 0] = f"{string.ascii_uppercase[n_less_0]}. [0]"
    output[x > 0] = pretty_qcut(x[x > 0], bins=bins_more_0, round_to=round_to, offset_letter=n_less_0+1)
    return output


def summary_statistics(df, category:str=None) -> pd.DataFrame:
    """gets all columns statistics for a class of data (numeric 
    or categorical)

    Args:
        category (str): 'numeric' or 'categorical'

    Returns:
        pd.DataFrame: summary of statistics
    """

    def get_mode(s, least_frequent=False) -> str:
        table_frequency = s.value_counts(ascending=least_frequent)
        return f"{table_frequency.index[0]} ({table_frequency.iloc[0]})"

    def stats_numeric(s) -> dict:
        return s.describe().to_dict()

    def stats_date(s) -> dict:
        return s.describe(datetime_is_numeric=True).to_dict()

    def stats_categorical(s) -> dict:
        # logging.debug(s[0:3])
        return {
            'Count': s.count(),
            '% NA': f"{s.isna().sum()/s.shape[0]:.2%}",
            'Distinct values': s.nunique(),
            'Most frequent value': get_mode(s),
            'Least frequent value': get_mode(s, least_frequent=True),
        }

    def apply_stats_func(df, func) -> pd.DataFrame:
        return df.apply(func, result_type='expand').T

    if category is None:
        summary_statistics(df, 'numeric')
        summary_statistics(df, 'categorical')
        summary_statistics(df, 'date')

    cols_numeric = df.select_dtypes([np.number]).columns
    cols_date = df.select_dtypes(['datetime']).columns
    cols_categorical = list(set(df.columns).difference(set(cols_date).union(set(cols_numeric))))
    if category == 'categorical':
        # cols_categorical = df.
        return apply_stats_func(df[cols_categorical], stats_categorical)
        # return df.dtypes
    elif category == 'numeric':
        return apply_stats_func(df[cols_numeric], stats_numeric)
    elif category == 'date':
        return apply_stats_func(df[cols_date], stats_date)
    elif category == 'all':
        return pd.concat([
            apply_stats_func(df[cols_numeric], stats_numeric),
            apply_stats_func(df[cols_date], stats_date),
            apply_stats_func(df[cols_categorical], stats_categorical),
        ])


def test_pandas_module():
    print("pandas_module_is_loaded")


def value_counts_and_normalize(s:pd.Series, val_name:str = None, 
                            percentage_100=False, round=4) -> pd.DataFrame:
    # Extends conventional pd.Series.value_counts() to show percentage and 
    # frequency at once    
    # percentage_100: if True, the percentage will be calculated based on 100
    #   instead of the total number of values
    # round: number of decimal places to round the percentage
    count = s.value_counts(dropna=False)
    if percentage_100:
        perc = (s.value_counts(dropna=False, normalize=True) * 100).round(round)
        columns = ['frequency', 'percentage (in %)']
    else:
        perc = (s.value_counts(dropna=False, normalize=True)).round(round)
        columns = ['frequency', 'proportion']
    output = pd.concat([count,perc], axis=1,
        keys=columns)
    if val_name is not None:
        output =  output.rename_axis(val_name).reset_index()
    return output


def value_counts_others(s:pd.Series, min_rank:int=5, others_label:str='Others', **kwargs) -> pd.DataFrame: 
    value_counts = s.value_counts()
    # Get the threshold based on the minimum rank
    threshold = value_counts.iloc[min_rank-1]
    # Get the infrequent values
    infrequent_values = value_counts[value_counts < threshold]
    # Replace the infrequent values with "Others"
    s = s.replace(infrequent_values.index, 'Others')
    # Get the new value counts
    new_value_counts = s.value_counts(**kwargs)    
    # Sort the new value counts by the value counts and the "Others" category
    new_value_counts = new_value_counts.sort_values(ascending=False)
    if others_label in new_value_counts.index:
        others_count = new_value_counts.pop(others_label)
        new_value_counts = new_value_counts.append(pd.Series(others_count, index=[others_label]))
    return new_value_counts
