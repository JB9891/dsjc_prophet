# dsjc_prophet
Repo for code used in Data Science Journal Club talk. The talk 
is about the Prophet package for time series forecasting.
The code of this repo can be used to better understand how prophet creates
its forecasts.

The rebuild with pymc is based on the following blog post:
- https://www.ritchievink.com/blog/2018/10/09/build-facebooks-prophet-in-pymc3-bayesian-time-series-analyis-with-generalized-additive-models/


## This repo has the following main parts
- `app_dsjc.py` - streamlit app to interactively see how the individual model 
parts are built and see how their parameters can be fit.
- `prophet_rebuild.py` - some helper functions used in the streamlit app and/or in the notebook
- `dsjc_pymc_rebuild.ipynb` - notebook with rebuild of prophet with pymc
- `linear_regression_with_pymc.py` - linear regression the bayesian way

## To run the app
- Install the needed packages with `poetry insatll` or `pip install -r requirements.txt`
- Run the app with `streamlit run app_dsjc.py`
