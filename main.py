import streamlit as st
import pandas as pd
import re
import mne
import os
import tempfile
from ml import pipeline, array_of_useful_channels as useful_ch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


st.set_option('deprecation.showPyplotGlobalUse', False)
logo_img = '/Users/kristianbogdan/PycharmProjects/DFO_2024/Tsao17DfQbg 2.png'
st.set_page_config("Диагностика НДС", logo_img)


# Preprocessing file
def file_ppc(f):
    pattern = r'(\d+\.\d+\.\d+):(\d+):(\d+)'
    line = f.read().decode('latin')
    f.seek(0)
    info = re.sub(pattern, r'\1.\2.\3', line).encode('latin')
    f.write(info)
    with tempfile.NamedTemporaryFile(prefix='signals', suffix='.EDF') as temp_file:
        temp_file.write(info)
        raw = mne.io.read_raw_edf(temp_file.name, preload=True)
        temp_file.close()
    f.close()
    del f
    return raw


st.logo(logo_img)
st.title("<- Для диагностики НДС подгрузите файл в формате .REC <-")


def create_dashboard(raw, ans, dots, ch,pred):
    if ans:
        st.header(f"У пациента было выявлено нарушение дыхания сна (HI - {pred})")
    else:
        st.header(f"У пациента не было выявлено нарушение дыхания сна")
    fig = raw.compute_psd().plot()
    st.pyplot(fig)
    del fig
    col1, col2, col3 = st.columns(3)

    with col1:
        for s in range(len(dots)//3):
            plt.plot(dots[s])
            plt.title(f'График сигнала {ch[s]}')
            plt.xlabel('Время, сек')
            plt.ylabel('ЭЭГ')
            st.pyplot()

    with col2:
        for s in range(len(dots)//3,len(dots)//3*2):
            plt.plot(dots[s])
            plt.title(f'График сигнала {ch[s]}')
            plt.xlabel('Время, сек')
            plt.ylabel('ЭЭГ')
            st.pyplot()


    with col3:
        for s in range(len(dots)//3*2,len(dots)):
            plt.plot(dots[s])
            plt.title(f'График сигнала {ch[s]}')
            plt.xlabel('Время, сек')
            plt.ylabel('ЭЭГ')
            st.pyplot()
        fig = raw.plot(picks=ch)
        st.pyplot(fig)



with st.sidebar:
    uploaded_file = st.file_uploader("Загрузите REC файл", type="REC")
if uploaded_file is not None:
    with st.sidebar:
        with st.spinner('Подождите, ЭЭГ анализируется...'):
            raw = file_ppc(uploaded_file)
            ans, dots, prediction = pipeline(raw)  # 0-не болен, 1-болен
            dots = dots[0]
            all_ch = set(raw.ch_names)
            itog_ch = list(set(useful_ch) & all_ch)
            st.success('Успешно!')

    create_dashboard(raw, ans, dots, itog_ch,prediction)

    # Твоя функция
