from setuptools import setup, find_packages, Extension, Command

setup(
    name="ML4qf",
    #version=__version__,
    description="""Machine Learning for Quantitative Finance""",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    keywords="S",
    author="Alvaro Cea",
    author_email="alvaro_cea@outlook.com",
    url="https://github.com/ACea15/ML4qf",
    license="",
    packages=find_packages(
        where='./',
        include=['ml4qf*'],
        ),
    # data_files=[
    #     ("./lib/UVLM/lib", ["libuvlm.so"]),
    #     ("./lib/xbeam/lib", ["libxbeam.so"])
    #     ],
    python_requires=">=3.10",
    install_requires=[
        "MiniSom",
        "QuantStats",
        "beautifulsoup4",
        "cufflinks",
        "getFamaFrenchFactors",
        "h5py",
        "jax",
        "jaxlib",
        "kaleido",
        "keras-tuner",
        "missingno",
        "networkx",
        "numpy",
        "pandas_ta",
        "plotly",
        "pydot",
        "pyfolio-reloaded",
        "requests",
        "scikeras",
        "scipy",
        "seaborn",
        "shap",
        "statsmodels",
        "tabulate",
        "tensorboard",
        "tensorboard",
        "tensorflow",
        "umap",
        "xgboost",
        "xlrd",
        "yfinance",
    ])
