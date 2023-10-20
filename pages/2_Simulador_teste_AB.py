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
    page_title="Inteli | Simulador de testes A/B",
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



if __name__ == '__main__':

    # Main page
    add_banner()

    st.write(    """
    # Inteli Exec Módulo 4 - Experimentação
    
    Bem-vindo(a)! Neste site, você poderá simular a construção de testes A/B.
    """)
   
    st.write("🚧  **Página em construção**")