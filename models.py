import statsmodels.api as sm
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class LinearReg(BaseEstimator, RegressorMixin):
    """
    Implements the statsmodels OLS regressor in a scikit-learn friendly way
    """

    def __init__(self):
        self.model = None
        self.results = None
        self.features = None
        self.pvalues = None
        self.feature_importances_ = None
        self.is_fit = False

    def fit(self, X, y):
        X_ = sm.add_constant(X)
        self.model = sm.OLS(y, X_)
        self.results = self.model.fit()
        self.features = self.model.exog_names
        self.pvalues = self.results.pvalues
        self.params = self.results.params
        self.feature_importances_ = -np.log10(self.pvalues)
        self.is_fit = True

    def __sklearn_is_fitted__(self):
        return self.is_fit

    def predict(self, X):
        X_ = sm.add_constant(X)
        return self.model.predict(self.results.params, X_)

    
class LogReg(BaseEstimator, ClassifierMixin):
    """
    Implements the statsmodels Logit regressor in a scikit-learn friendly way
    """

    def __init__(self):
        self.model = None
        self.results = None
        self.features = None
        self.pvalues = None
        self.feature_importances_ = None
        self.is_fit = False


    def fit(self, X, y):
        X_ = sm.add_constant(X)
        self.classes_ = np.unique(y)
        self.model = sm.Logit(y, X_)
        self.results = self.model.fit();
        self.pvalues = self.results.pvalues
        self.params = self.results.params
        self.feature_importances_ = -np.log10(self.pvalues)
        self.is_fit = True
        return self

    def __sklearn_is_fitted__(self):
        return self.is_fit

    def predict_proba(self, X):
        X_ = sm.add_constant(X)
        y_probs= self.model.predict(self.results.params, X_)
        return np.array([1-y_probs,y_probs]).T
    
    def predict(self, X):
        preds = self.predict_proba(X)
        return np.where(preds[:,1] >= 0.5, 1, 0)

def load_regression_model(model_type):
    if model_type == 'Regressão linear':
        return LinearReg()
    elif model_type == 'Random Forest':
        return RandomForestRegressor(random_state=123, max_depth=3)


def load_classification_model(model_type):
    if model_type == 'Regressão logística':
        return LogReg()

    elif model_type == 'Random Forest':
        return RandomForestClassifier(random_state=201, max_depth=3, class_weight='balanced')


def model_is_fit(model):
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError
    is_fit = False
    try:
        is_fit = check_is_fitted(model) is None
    except NotFittedError:
        pass
    except TypeError:
        pass
    return is_fit


def instantiate_model():
    problem_type = st.selectbox(label="Escolha o tipo de problema a abordar",
                                options=["Selecionar...",
                                         "Regressão",
                                         "Classificação"],
                                index=0)

    if problem_type == "Regressão":

        model_type = st.selectbox(label="Escolha o algoritmo a ser usado",
                                  options=["Selecionar...",
                                           "Regressão linear",
                                           "Random Forest"],
                                  index=0)
        model = load_regression_model(model_type)

    elif problem_type == 'Classificação':
        model_type = st.selectbox(label="Escolha o algoritmo a ser usado",
                                  options=["Selecionar...",
                                           "Regressão logística",
                                           "Random Forest"],
                                  index=0)
        model = load_classification_model(model_type)
    else:
        model = None

    return model, problem_type
