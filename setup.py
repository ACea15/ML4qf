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
                "cufflinks",
                "h5py",
                "MiniSom",
                "missingno",
                "networkx",
                "plotly",
                "pydot",
                "pyfolio-reloaded",
                "QuantStats",
                "seaborn",
                "tabulate",
                "tensorboard",
                "shap",
                "xgboost",
                "xlrd",
                "yfinance",
                "keras-tuner",
                "tensorboard",
                "pandas_ta",
                "scikeras"
    ])
