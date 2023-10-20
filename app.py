# access this file from cd ~/Library/Mobile\ Documents/com~apple~CloudDocs/ in the Mac terminal
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import messages as msn
import models
import metrics as met

from PIL import Image


plt.style.use("dark_background")
mpl.rcParams['figure.dpi'] = 210
font = {'family': 'Tahoma', 'size': 14}
mpl.rc('font', **font)

# Configurations
st.set_page_config(
    page_title="Inteli | Modelos online",
    page_icon="💻",
    layout="wide",
    menu_items={
        "Get help": "mailto:Alessandro.Gagliardi@br.experian.com",
        "About": """Página construída para curso de dados do Inteli (2023)"""
    }
)


def add_banner():
    banner = Image.open(os.path.join('assets', 'inteli_logo.png'))
    st.image(banner)
    return


def get_features_and_target(data):
    X, y, run_training = None, None, None
    if data is None:
        pass
    else:
        col1, col2 = st.columns(2)
        df = pd.read_excel(data)
        col1.write("Visualização da sua base de dados")
        col1.dataframe(df)
        cols = df.columns.tolist()
        features = col2.multiselect(label="Escolha as covariáveis para treinar o modelo",
                                    options=cols, default=cols)
        target = col2.multiselect(
            label="Escolha o alvo para treinar o modelo. Seu alvo NÃO pode ser uma das covariáveis!",
            options=cols,
            max_selections=1)

        exclusive = set(features).intersection(target) == set()
        has_target = len(target) == 1
        if exclusive & has_target:
            col2.write("✅ Sua escolha de covariáveis e alvo estão boas. Pode ir em frente com o treino.")
            X, y = df[features], df[target[0]]
            run_training = col2.button("Rodar treino")
        else:
            col2.write("❌ Garanta que você escolheu apenas um alvo e que ele não faça parte das variáveis de treino!")
    return X, y, run_training


if __name__ == '__main__':

    # Main page
    add_banner()
    st.write(msn.welcome)

    # choose type of problem
    model, problem_type = models.instantiate_model()
    data, X, y = None, None, None
    if model is not None:
        # Load data
        #st.expander("Como devem estar formatados meus dados?").write(msn.data_upload_explanation)

        #data = st.file_uploader("Adicionar base com covariáveis e alvo para treino (.xlsx)")
        import os
        st.write(os.getcwd())
        if problem_type == 'Classificação':
            data = 'data/Módulo 3.2 - Bases processadas.xlsx'
        else:
            data = 'data/Módulo 3.1 - Bases processadas.xlsx'
            
        X, y, run_training = get_features_and_target(data)

        if run_training:
            model.fit(X, y)

    if models.model_is_fit(model):
        st.write("---")
        st.write("""# Diagnóstico do modelo""")

        col1, col2, col3 = st.columns(3)

        # TODO: refatorar código abaixo

        # Feature importances
        col1.write("### Importância das variáveis")

        fig, ax = plt.subplots()
        pd.Series(model.feature_importances_, index=X.columns). \
            sort_values(ascending=True). \
            plot(kind='barh', ax=ax)
        ax.axvline(1.301, linestyle='--', color='white')  # p = 0.05 threshold
        ax.set_xlabel("Feature importances")
        col1.pyplot(fig)
        col1.expander("Entendendo o cálculo").write(msn.feature_importance)

        # Coefficients and standard errors
        # TODO: isso aqui é um puxadinho
        try:
            res_df = pd.DataFrame(model.results.params, columns=['Coeficiente']).join(
                pd.DataFrame(model.results.bse, columns=['Erro padrão']).join(
                    pd.DataFrame(model.pvalues, columns=['p-valor'])
                )
            ).sort_values('p-valor')
            col2.write("### Coeficientes obtidos")
            col2.table(res_df)
        except AttributeError:
            pass

        # Metrics (adjusted R2, RMSE for regression; ROC AUC,

        col3.write("### Métricas")

        if problem_type == 'Classificação':
            y_probs = model.predict_proba(X)[:, 1]
            roc_auc = met.roc_auc(y, y_probs)
            ks = met.ks_score(y, y_probs)
            col3.metric(label="ROC AUC", value=f"{round(roc_auc, 4)}")
            col3.metric(label="KS Score", value=f"{round(ks, 2)}")

        elif problem_type == 'Regressão':
            y_pred = model.predict(X)
            r2_adj = met.adjusted_r2(y, y_pred, p=len(X.columns))
            rmse = np.sqrt(((y - y_pred) ** 2).mean())
            col3.metric(label="R2 adjustado", value=f"{round(r2_adj, 4)}")

            col3.metric(label="Root mean squared error", value=f"{round(rmse, 2)}")

        else:
            raise Exception
