from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mne
import numpy as np
import torch
from catboost import CatBoostRegressor

from .model import Chrononet


USEFUL_CHANNELS = [
    "Fp1-M2",
    "C3-M2",
    "O1-M2",
    "Fp2-M1",
    "C4-M1",
    "O2-M1",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_PATH = MODELS_DIR / "best_model.pth"
REGRESSOR_PATH = MODELS_DIR / "cb_model.cbm"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mne.set_log_level("ERROR")


@lru_cache(maxsize=1)
def get_classifier() -> Chrononet:
    model = Chrononet()
    state_dict = torch.load(CLASSIFIER_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_hi_regressor() -> CatBoostRegressor:
    regressor = CatBoostRegressor()
    regressor.load_model(str(REGRESSOR_PATH))
    return regressor


def read_data_for_inference(raw: mne.io.BaseRaw) -> np.ndarray:
    prepared_raw = raw.copy()
    prepared_raw.set_eeg_reference()
    prepared_raw.filter(l_freq=1, h_freq=45)

    useful_positions: list[int] = []
    raw_channel_indices: list[int] = []
    for position, channel_name in enumerate(USEFUL_CHANNELS):
        if channel_name in prepared_raw.ch_names:
            useful_positions.append(position)
            raw_channel_indices.append(prepared_raw.ch_names.index(channel_name))

    if not raw_channel_indices:
        raise ValueError(
            "В файле не найдены поддерживаемые каналы: "
            + ", ".join(USEFUL_CHANNELS)
        )

    epochs = mne.make_fixed_length_epochs(prepared_raw, duration=150, overlap=0)
    epoch_data = epochs.get_data(picks=raw_channel_indices).astype(np.float32)

    if epoch_data.shape[0] == 0:
        raise ValueError("Запись слишком короткая для анализа 150-секундных эпох.")

    aligned_epochs = np.zeros(
        (epoch_data.shape[0], len(USEFUL_CHANNELS), epoch_data.shape[2]),
        dtype=np.float32,
    )

    for source_index, target_index in enumerate(useful_positions):
        aligned_epochs[:, target_index, :] = epoch_data[:, source_index, :]

    return aligned_epochs


def predict(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    classifier = get_classifier()
    tensor = torch.from_numpy(data).to(DEVICE)

    with torch.no_grad():
        logits, embeddings = classifier(tensor, return_emb=True)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
        representative_index = int(
            torch.argmax(probabilities.max(dim=1).values).item()
        )

    return predicted_classes, embeddings.cpu().numpy(), representative_index


def predict_hi(embeddings: np.ndarray) -> float:
    return float(get_hi_regressor().predict(embeddings).mean())


def pipeline(raw: mne.io.BaseRaw) -> tuple[int, np.ndarray, float | None]:
    data = read_data_for_inference(raw)
    predictions, embeddings, representative_index = predict(data)
    positive_votes = int((predictions == 1).sum())
    negative_votes = int((predictions == 0).sum())
    representative_epoch = data[representative_index : representative_index + 1]

    if positive_votes >= negative_votes:
        return 1, representative_epoch, predict_hi(embeddings)

    return 0, representative_epoch, None
