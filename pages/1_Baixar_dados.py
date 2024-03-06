import os
import io
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
    page_title="Inteli | Baixar dados para treino de modelos",
    page_icon="💻",
    layout="wide",
    menu_items={
        "Get help": "mailto:Alessandro.Gagliardi@br.experian.com",
        "About": """Página construída para curso de dados do Inteli (2023)"""
    }
)

if __name__ == '__main__':
    # banner = Image.open(os.path.join('assets', 'inteli_logo.png'))
    # st.image(banner)

    st.write(
"""Use os botões abaixo para baixar os dados. 
             
Ambos os arquivos tem as mesmas variáveis, mas somente o de treino contém as marcações de `default` realizadas.

Seu objetivo será de estudar as variáveis no treino, construir variáveis novas se achar necessário, e realizar o treinamento. 
Com isso, poderá obter os scores (probabilidade de inadimplência) para a base de teste.
""")

    # @st.cache_data
    def load_dfs():
        train = pd.read_csv("data/processed/InteliBank_Inadimplencia_de_credito__Treino.csv")
        test = pd.read_csv("data/processed/InteliBank_Inadimplencia_de_credito__Avaliacao.csv")
        return train, test
    
    train, test = load_dfs()

    def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
        processed_data = output.getvalue()
        return processed_data
    
    # Verifica se o DataFrame já foi processado e o botão de download já foi criado
    if 'download_ready' not in st.session_state or not st.session_state.download_ready:
        with st.spinner('Aguarde... Preparando o arquivo para download.'):
            # Chamando a função to_excel para obter o Excel em formato de bytes
            train_bytes = to_excel(train)
            test_bytes = to_excel(test)

            st.session_state['train_bytes'] = train_bytes
            st.session_state['test_bytes'] = test_bytes
            st.session_state['download_ready'] = True

    # Sempre mostrar o botão de download após o processamento
    if 'download_ready' in st.session_state and st.session_state.download_ready:
        st.download_button(
            label="Download Train set",
            data=st.session_state.train_bytes,
            file_name="base_treino.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.download_button(
            label="Download Test set",
            data=st.session_state.test_bytes,
            file_name="base_test.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
