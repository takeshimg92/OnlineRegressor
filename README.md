# OnlineRegressor
Streamlit Cloud app for no-code regression and classification tasks.
This code was developed to be used as a demonstration tool in lectures about data science for managers who do not have a background in analytics.

Datasets are expected to be uploaded in runtime, as `.xlsx` files. There is no split (as far as Aug 2023) between training and testing, so the Random Forest option (which is prone to overfitting) has a hard coded `max_depth = 3`. 

