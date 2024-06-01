import streamlit as st
import re
import mne
import os
import tempfile
from ml import pipeline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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

        ans, dots, prediction = pipeline(raw)
        dots = dots[0]
        print(dots[0][0], dots[1][0], dots[2][0], dots[3][0], dots[4][0], dots[5][0])
        for s in range(len(dots)):
            plt.plot(dots[s])
            plt.title('График сигнала')
            plt.xlabel('Время, сек')
            plt.ylabel('ЭЭГ')
            #plt.show()
            plt.savefig(f'channel{s}.png')

ans=''
st.text(ans)
