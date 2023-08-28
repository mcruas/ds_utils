from setuptools import setup, find_packages

setup(
    name="ds_utils",
    version="0.0.1",
    url="https://github.com/mcruas/ds_utils",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
    ]
)