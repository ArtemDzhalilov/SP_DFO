import streamlit as st
import re
import mne
import os
import tempfile
from ml import pipeline


# Preprocessing file
def file_ppc(f):
    pattern = r'(\d+\.\d+\.\d+):(\d+):(\d+)'
    line = f.read().decode('latin')
    f.seek(0)
    info = re.sub(pattern, r'\1.\2.\3', line).encode('latin')
    f.write(info)
    new_file_path = 'signals.EDF'
    with open(new_file_path, 'wb') as new_file:
        new_file.write(info)
        raw = mne.io.read_raw_edf(new_file_path, preload=True)
    f.close()
    del f
    return raw


st.title("Интерфейс загрузки и обработки файла")

with st.sidebar:
    uploaded_file = st.file_uploader("Загрузите EDF файл", type="REC")
    if uploaded_file is not None:
        raw = file_ppc(uploaded_file)

        #Твоя функция
        ans, dots, prediction = pipeline(raw)
ans=''
st.text(ans)