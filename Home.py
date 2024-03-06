import streamlit as st
from PIL import Image
import os

st.set_page_config(
    page_title="Inteli Exec",
    page_icon="👋",
)
banner = Image.open(os.path.join('assets', 'inteli_logo.png'))
st.image(banner)

st.write("# Inteli Exec - Treinamento de modelos de machine learning")

#st.sidebar.success("Selecione .")

st.markdown(
    """
    Esta página contém conteúdos específicos para o Módulo 3:
    
    * Dados para download
    * Interface para treino de modelos de machine learning


"""
)