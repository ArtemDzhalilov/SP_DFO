from __future__ import annotations

import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import streamlit as st

from src.sp_dfo import USEFUL_CHANNELS, pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
LOGO_PATH = PROJECT_ROOT / "assets" / "logo.png"
PAGE_TITLE = "SomnoScope"
TIME_PATTERN = re.compile(r"(\d+\.\d+\.\d+):(\d+):(\d+)")

mne.set_log_level("ERROR")

st.set_page_config(
    page_title="SomnoScope | Диагностика НДС",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else None,
    layout="wide",
)


def preprocess_uploaded_record(uploaded_file) -> mne.io.BaseRaw:
    normalized_bytes = TIME_PATTERN.sub(
        r"\1.\2.\3",
        uploaded_file.getvalue().decode("latin-1"),
    ).encode("latin-1")

    with tempfile.NamedTemporaryFile(
        prefix="sp_dfo_",
        suffix=".edf",
        delete=False,
    ) as temp_file:
        temp_file.write(normalized_bytes)
        temp_path = Path(temp_file.name)

    try:
        return mne.io.read_raw_edf(temp_path, preload=True)
    finally:
        temp_path.unlink(missing_ok=True)


def plot_psd(raw: mne.io.BaseRaw) -> None:
    fig = raw.compute_psd().plot(show=False)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def plot_representative_epoch(raw: mne.io.BaseRaw, epoch) -> None:
    available_channels = set(raw.ch_names)
    columns = st.columns(3)

    for index, channel_name in enumerate(USEFUL_CHANNELS):
        title = channel_name
        if channel_name not in available_channels:
            title = f"{channel_name} (нет в записи)"

        fig, ax = plt.subplots(figsize=(5.5, 2.2))
        ax.plot(epoch[index], linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Отсчеты")
        ax.set_ylabel("Амплитуда")
        ax.grid(alpha=0.2)
        columns[index % 3].pyplot(fig, clear_figure=True)
        plt.close(fig)


def render_header() -> None:
    logo_column, text_column = st.columns([1, 4])

    with logo_column:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)

    with text_column:
        st.title(PAGE_TITLE)
        st.write(
            "Платформа для предварительного анализа REC-записей "
            "и выявления признаков нарушений дыхания сна по ЭЭГ."
        )
        st.caption(
            "Поддерживаемые каналы: " + ", ".join(USEFUL_CHANNELS)
        )


def render_result(raw: mne.io.BaseRaw, has_disorder: bool, epoch, hi_score) -> None:
    result_text = "НДС выявлено" if has_disorder else "Признаки НДС не выявлены"

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Статус", result_text)
    metric_2.metric(
        "HI",
        f"{hi_score:.2f}" if hi_score is not None else "не рассчитывался",
    )
    metric_3.metric(
        "Доступных каналов",
        str(sum(channel in raw.ch_names for channel in USEFUL_CHANNELS)),
    )

    st.subheader("Спектральный анализ записи")
    plot_psd(raw)

    st.subheader("Репрезентативный фрагмент сигнала")
    plot_representative_epoch(raw, epoch)


render_header()

with st.sidebar:
    st.header("Загрузка данных")
    uploaded_file = st.file_uploader(
        "Выберите REC-файл пациента",
        type=["REC", "rec"],
    )
    st.caption(
        "Модель использует фиксированные 150-секундные эпохи и шесть "
        "ключевых ЭЭГ-каналов."
    )

if uploaded_file is None:
    st.info("Загрузите REC-файл, чтобы запустить анализ.")
else:
    try:
        with st.spinner("Файл обрабатывается, модель выполняет инференс..."):
            raw = preprocess_uploaded_record(uploaded_file)
            has_disorder, representative_epoch, hi_score = pipeline(raw)

        render_result(raw, bool(has_disorder), representative_epoch[0], hi_score)
    except Exception as error:
        st.error(f"Не удалось обработать файл: {error}")
