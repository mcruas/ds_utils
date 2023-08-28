import logging
from pathlib import Path
from typing import Dict
import polars as pl
import pandas as pd
import os
from jinja2 import Template


def get_data(query, file_path, force_query=False, save=True, project_id=None, **kwargs):
    # get data from bigquery if it is not saved locally
    # if file_path is of .csv type, than save using .to_csv instead of .to_parquet
        
    file_path = os.path.join(os.getcwd(), file_path)
    if not os.path.exists(file_path) or force_query:
        query_run = Template(query).render(**kwargs)
        df = pd.read_gbq(query_run, project_id=project_id, dialect='standard')
        if save: df.to_parquet(file_path)
    else:
        df = pd.read_parquet(file_path)
    return df

def test_io_module():
    print("io_module_is_loaded")

# TODO Test the following functions and remove them if they are not needed

# # wrapper to choose which function to use to get data
# def get_data(*args, use_polars=False, **kwargs):
#     if use_polars:
#         return get_data_pl(*args, **kwargs)
#     else:
#         return get_data_pd(*args, **kwargs)


# def run_query(query, project_id=None, use_polars=False):
#     from google.cloud import bigquery
#     client = bigquery.Client(project_id)
#     logging.info('Querying BQ...')
#     query_job = client.query(query)
#     if use_polars:
#         return pl.from_arrow(query_job.result().to_arrow())
#     else:
#         return query_job.to_dataframe()


# def read_gbq_from_template(template_query, dict_query = {}, 
# project_id=, use_polars=False):
#     # Reads a query from a template and returns the query with the variables replaced
#     # template_query: query as string, may use jinja2 templating
#     # dict_query: dictionary of query parameters, to render from the template with jinja2
#     query = Template(template_query).render(dict_query)
#     logging.info('Getting dataset from BQ...')
#     # return pd.read_gbq(query, progress_bar_type='tqdm',
#     #     project_id=project_id)
#     return run_query(query, project_id=project_id, use_polars=use_polars)


# def get_data_pd(file_path, template_query, force_query=False, make_dir=True ,
# only_file=False, dict_query={}, save=True, project_id='dataplatform-prd',
# **kwargs):
#     # Get data either from file_path or from BQ
#     # file_path: path to file
#     # template_query: query as string, may use jinja2 templating
#     # force_query: force query to be executed
#     # make_dir: make directory if it doesn't exist
#     # only_file: if file doesn't exist, return FileNotFoundError. Ignored if force_query is True
#     # dict_query: dictionary of query parameters, to render from the template with jinja2
#     # kwargs: additional parameters to pass to pandas.read functions

#     def make_dir_if_not_exists(fn):
#         # Make directory if it doesn't exist
#         # fp: filepath
#         if fn.startswith('s3://'):
#             pass
#         elif not (fp := Path(fn).parent.exists()):
#             fp.parent.mkdir(parents=True)

#     def make_dir_if_not_exists(fn):
#         # Make directory if it doesn't exist
#         # fp: filepath
#         if not (fp := Path(fn).parent.exists()):
#             fp.parent.mkdir(parents=True)
    
#     def read_file_with_pandas(fn, **kwargs):
#         # Read file with pandas
#         # fn: filepath
#         # kwargs: additional parameters to pass to pandas.read functions
#         extension = fn.split('.')[-1]
#         if extension == 'csv':
#             return pd.read_csv(fn, **kwargs)
#         elif extension == 'xlsx':
#             return pd.read_excel(fn, **kwargs)
#         elif extension == 'parquet':
#             # import pyarrow as pa
#             # custom_dict = {'dbdate': pa.date32()}
#             # table = pa.parquet.read_table(fn, **kwargs)
#             # return table.to_pandas()
#             return pd.read_parquet(fn, engine="fastparquet", **kwargs)
#         elif extension == 'pickle':
#             return pd.read_pickle(fn, **kwargs)

#     def save_file_with_pandas(df, fn, **kwargs):
#         # Save file with pandas
#         # df: dataframe
#         # fn: filepath
#         # kwargs: additional parameters to pass to pandas.read functions
#         extension = fn.split('.')[-1]
#         if extension == 'csv':
#             df.to_csv(fn, **kwargs)
#         elif extension == 'xlsx':
#             df.to_excel(fn, **kwargs)
#         elif extension == 'parquet':
#             df.to_parquet(fn, **kwargs)
#         elif extension == 'pickle':
#             df.to_pickle(fn, **kwargs)

#     fn = convert_filepath_s3(file_path)
#     if make_dir: 
#         make_dir_if_not_exists(fn)
#     if not force_query:
#         try: # try to read from file
#             df = read_file_with_pandas(fn, **kwargs)
#             logging.info(f"File found on path: {file_path}. Reading from file...")
#             return df
#         except: # if file not found, query BQ
#             if only_file:
#                 raise FileNotFoundError(f"File not found on path: {fn}")
#             logging.info(f"File not found on path: {fn}. Querying BQ...")
#             df = read_gbq_from_template(template_query, dict_query, 
#                                         project_id=project_id)            
#     else: # if force_query is True 
#         logging.info(f'Force query is True. Querying BQ...')
#         df = read_gbq_from_template(template_query, dict_query, 
#                                     project_id=project_id)            
#     # check file extension and save it accordingly
#     if save:
#         save_file_with_pandas(df, fn, **kwargs)
#         logging.info(f"Saving file to {fn}")
#     return df



# def get_data_pl(file_path: str, template_query: str, force_query: bool = False, 
# make_dir: bool = True, only_file: bool = False, dict_query: Dict = {}, 
# save: bool = True, project_id: str = , **kwargs) -> pl.DataFrame:
#     """
#     Get data either from a file or from BigQuery, and optionally save it to the file.

#     Parameters:
#     -----------
#     file_path: str
#         Path to file.
#     template_query: str
#         Query as string, may use jinja2 templating.
#     force_query: bool
#         Force query to be executed.
#     make_dir: bool
#         Make directory if it doesn't exist.
#     only_file: bool
#         If file doesn't exist, raise FileNotFoundError. Ignored if force_query is True.
#     dict_query: dict
#         Dictionary of query parameters to render from the template with jinja2.
#     save: bool
#         Save the DataFrame to the file after reading it.
#     project_id: str
#         ID of the BigQuery project to execute the query on.
#     kwargs: dict
#         Additional parameters to pass to the Pandas read functions.

#     Returns:
#     --------
#     pl.DataFrame
#         DataFrame containing the query results.
#     """    

#     def make_dir_if_not_exists(fn):
#         path_parent = Path(fn).parent
#         # Path(fn).parent
#         if not (path_parent.exists()):
#             path_parent.mkdir(parents=True)
    
#     def read_file_with_polars(fn, **kwargs):
#         extension = fn.split('.')[-1]
#         if extension == 'csv':
#             return pl.read_csv(fn, **kwargs)
#         elif extension == 'xlsx':
#             return pl.read_excel(fn, **kwargs)
#         elif extension == 'parquet':
#             return pl.read_parquet(fn, **kwargs)
#         elif extension == 'pickle':
#             # read with pandas and convert to polars
#             return pl.from_pandas(pd.read_pickle(fn, **kwargs))
#         else:
#             raise ValueError(f"File extension {extension} not supported by polars.")
        
#     def save_file_with_polars(df, fn, **kwargs):
#         extension = fn.split('.')[-1]
#         if extension == 'csv':
#             df.write_csv(fn, **kwargs)
#         elif extension == 'xlsx':
#             df.write_excel(fn, **kwargs)
#         elif extension == 'parquet':
#             df.write_parquet(fn, **kwargs)
#         elif extension == 'pickle':
#             # convert to pandas and save
#             df.to_pandas().to_pickle(fn, **kwargs)

#     if make_dir: 
#         make_dir_if_not_exists(file_path)
#     if not force_query:
#         try: # try to read from file
#             df = read_file_with_polars(file_path, **kwargs)
#             logging.info(f"File found on path: {file_path}. Reading from file...")
#             return df
#         except: # if file not found, query BQ
#             if only_file:
#                 raise FileNotFoundError(f"File not found on path: {file_path}")
#             logging.info(f"File not found on path: {file_path}. Querying BQ...")
#             df = read_gbq_from_template(template_query, dict_query, 
#                                         project_id=project_id, use_polars=True)            
#     else: # if force_query is True 
#         logging.info(f'Force query is True. Querying BQ...')
#         df = read_gbq_from_template(template_query, dict_query, 
#                                     project_id=project_id, use_polars=True)            
#     # check file extension and save it accordingly
#     if save:
#         save_file_with_polars(df, file_path, **kwargs)
#         logging.info(f"Saving file to {file_path}")
#     return df
