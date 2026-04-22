from __future__ import annotations

import argparse
import re
from pathlib import Path


TIME_PATTERN = re.compile(r"(\d+\.\d+\.\d+):(\d+):(\d+)")


def normalize_header(file_path: Path) -> None:
    content = file_path.read_text(encoding="latin-1")
    normalized = TIME_PATTERN.sub(r"\1.\2.\3", content)
    file_path.write_text(normalized, encoding="latin-1")


def convert_rec_to_edf(file_path: Path) -> Path:
    edf_path = file_path.with_suffix(".EDF")
    if file_path != edf_path:
        file_path.rename(edf_path)
    normalize_header(edf_path)
    return edf_path


def find_rec_files(root: Path) -> list[Path]:
    uppercase = list(root.rglob("*.REC"))
    lowercase = list(root.rglob("*.rec"))
    return sorted({*uppercase, *lowercase})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготовка REC-файлов: переименование в EDF и нормализация заголовков.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Папка с REC-файлами или корневая директория датасета.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.path.expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Путь не найден: {source_path}")

    rec_files = [source_path] if source_path.is_file() else find_rec_files(source_path)
    if not rec_files:
        print("REC-файлы не найдены.")
        return

    for rec_file in rec_files:
        converted_file = convert_rec_to_edf(rec_file)
        print(converted_file)

    print(f"\nПодготовлено файлов: {len(rec_files)}")


if __name__ == "__main__":
    main()
