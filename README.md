# Package `ds_utils`
Miscelaneous tools for data science applications. The goal of this package is to provide a set of tools that are useful for data science applications, but are not specific to any particular domain.

The package is currently in early development, and is not yet available on PyPI.

The tools are organized into submodules, which are:

- **io**: Tools for reading and writing data.
- **misc**: Miscelaneous tools.
- **pandas_utils**: Tools for working with pandas DataFrames.
- **plot_utils**: Tools for plotting data.
- **polars_utils**: Tools for working with polars DataFrames.


## Installation

To install, use the following command:

```bash
pip install git+https://github.com/mcruas/ds_utils.git@v0.0.1
```

## Usage

```python
from ds_utils import io, misc

# read a query from a text file
query = misc.read_text("queries\evaluator_data.sql")
# if the file already exists, load it from disk. Otherwise, run the query, save the result to disk and return it.
df = io.get_data(query, "data\evaluator_data.parquet", force_query=False, project_id="my_project", save=True)
```
