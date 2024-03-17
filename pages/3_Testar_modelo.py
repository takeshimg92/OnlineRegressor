import streamlit as st
import pandas as pd

if  st.session_state.get('is_fit') is None:
    st.write("Você precisa treinar um modelo antes")
    st.write("### Importante! \n As features devem ser *exatamente* as mesmas usadas para treino")
else:
    if st.session_state['is_fit']:
        st.write("# Aplicar score na base de teste")
        val = st.file_uploader("Envie seu arquivo para teste", type=['xlsx'], accept_multiple_files=False)
        if val is not None:
            model = st.session_state['model']
            df_val = pd.read_excel(val)
            X_val = df_val[st.session_state['features']]
            df_pred = X_val.copy()
            df_pred['Prediction'] = model.predict_proba(X_val)[:,1]
            st.write("É possível fazer download desta tabela clicando no símbolo no canto direito superior:")
            st.dataframe(df_pred)

