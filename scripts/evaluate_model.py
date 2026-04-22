from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sp_dfo.inference import read_data_for_inference  # noqa: E402
from src.sp_dfo.model import Chrononet  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_PATTERN = re.compile(r"(\d+\.\d+\.\d+):(\d+):(\d+)")
mne.set_log_level("ERROR")


class EpochDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[np.ndarray, int]],
        apply_masks: bool = False,
        zero_out_probability: float = 0.5,
    ) -> None:
        self.samples = samples
        self.apply_masks = apply_masks
        self.zero_out_probability = zero_out_probability

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        epoch, label = self.samples[index]
        epoch_tensor = torch.tensor(epoch, dtype=torch.float32)

        if self.apply_masks:
            for channel_index in range(epoch_tensor.shape[0]):
                if random.random() < self.zero_out_probability:
                    epoch_tensor[channel_index] = torch.zeros_like(epoch_tensor[channel_index])

        return epoch_tensor, torch.tensor(label, dtype=torch.long)


def convert_rec_to_edf(file_path: Path) -> Path:
    edf_path = file_path.with_suffix(".EDF")
    if file_path != edf_path:
        file_path.rename(edf_path)

    content = edf_path.read_text(encoding="latin-1")
    normalized = TIME_PATTERN.sub(r"\1.\2.\3", content)
    edf_path.write_text(normalized, encoding="latin-1")
    return edf_path


def resolve_record_path(path: Path) -> Path:
    if path.is_file():
        if path.suffix.lower() == ".edf":
            return path
        if path.suffix.lower() == ".rec":
            return convert_rec_to_edf(path)
        raise ValueError(f"Неподдерживаемый формат файла: {path}")

    rec_candidates = sorted(list(path.glob("*.REC")) + list(path.glob("*.rec")))
    if rec_candidates:
        return convert_rec_to_edf(rec_candidates[0])

    edf_candidates = sorted(list(path.glob("*.EDF")) + list(path.glob("*.edf")))
    if edf_candidates:
        return edf_candidates[0]

    raise FileNotFoundError(f"Не найден REC/EDF-файл в {path}")


def infer_label(path: Path) -> int:
    match = re.search(r"Nr\s*([0-9]+)", str(path), re.IGNORECASE)
    if not match:
        raise ValueError(
            "Не удалось определить метку класса по пути. "
            "Ожидается шаблон вроде 'Nr 1' или 'Nr 2'."
        )
    return 0 if match.group(1) == "1" else 1


def load_samples(paths: list[Path]) -> list[tuple[np.ndarray, int]]:
    samples: list[tuple[np.ndarray, int]] = []

    for path in paths:
        record_path = resolve_record_path(path.expanduser().resolve())
        raw = mne.io.read_raw_edf(record_path, preload=True)
        epochs = read_data_for_inference(raw)
        label = infer_label(path)
        samples.extend((epoch, label) for epoch in epochs)

    return samples


def load_model() -> Chrononet:
    model = Chrononet()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def collect_predictions(
    model: Chrononet,
    dataloader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            logits = model(inputs.to(DEVICE))
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            labels.extend(targets.numpy().tolist())

    return np.array(predictions), np.array(labels)


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    accuracy = float((predictions == labels).mean())
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценка классификатора Chrononet на наборе директорий или EDF/REC-файлов.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Пути до директорий с записями или конкретных EDF/REC-файлов.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Размер батча для инференса.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.paths)
    if not samples:
        raise RuntimeError("Не удалось собрать датасет для оценки.")

    model = load_model()

    plain_loader = DataLoader(
        EpochDataset(samples, apply_masks=False),
        batch_size=args.batch_size,
        shuffle=False,
    )
    masked_loader = DataLoader(
        EpochDataset(samples, apply_masks=True),
        batch_size=args.batch_size,
        shuffle=False,
    )

    plain_predictions, plain_labels = collect_predictions(model, plain_loader)
    masked_predictions, masked_labels = collect_predictions(model, masked_loader)

    print_metrics("Без маскирования каналов", calculate_metrics(plain_predictions, plain_labels))
    print_metrics("С маскированием каналов", calculate_metrics(masked_predictions, masked_labels))


if __name__ == "__main__":
    main()
