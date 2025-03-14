import setuptools

__version__ = "0.0.0"

REPO_NAME = "mlops-walmart-forecast-mlops"
AUTHOR_USER_NAME = "krsx"
SRC_REPO = "walmart_sales_forecasting"
AUTHOR_EMAIL = "krisnaerlangga08@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Python package for the ML application",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
