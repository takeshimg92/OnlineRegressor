import streamlit as st
from PIL import Image
import os

st.set_page_config(
    page_title="Inteli Exec",
    page_icon="👋",
)
banner = Image.open(os.path.join('assets', 'inteli_logo.png'))
st.image(banner)

st.write("# Inteli Exec - Aplicações")

#st.sidebar.success("Selecione .")

st.markdown(
    """
    Esta página contém:
    
    * Modelos de regressão & classificação (Módulo 3)
    * Simulador de teste A/B (Módulo 4)
"""
)