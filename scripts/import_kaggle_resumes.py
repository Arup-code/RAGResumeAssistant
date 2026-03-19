from __future__ import annotations

import csv
import shutil
from pathlib import Path

import kagglehub

DATASET_ID = "palaksood97/resume-dataset"
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
TEXT_COLUMN_HINTS = {
    "resume",
    "resume_text",
    "resume_str",
    "text",
    "content",
    "description",
}


def _pick_text_column(fieldnames: list[str]) -> str | None:
    normalized = {name.lower(): name for name in fieldnames}
    for hint in TEXT_COLUMN_HINTS:
        if hint in normalized:
            return normalized[hint]
    return None


def _copy_supported_files(dataset_root: Path, target_dir: Path) -> int:
    copied = 0
    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        target_name = f"kaggle_{file_path.stem}{file_path.suffix.lower()}"
        destination = target_dir / target_name
        counter = 1
        while destination.exists():
            destination = target_dir / f"kaggle_{file_path.stem}_{counter}{file_path.suffix.lower()}"
            counter += 1

        shutil.copy2(file_path, destination)
        copied += 1
    return copied


def _extract_resumes_from_csv(dataset_root: Path, target_dir: Path) -> int:
    extracted = 0
    for csv_file in dataset_root.rglob("*.csv"):
        with csv_file.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue

            text_column = _pick_text_column(reader.fieldnames)
            for idx, row in enumerate(reader, start=1):
                if text_column and row.get(text_column):
                    resume_text = row[text_column].strip()
                else:
                    # Fallback: use the longest column value as best-effort resume text.
                    values = [value.strip() for value in row.values() if isinstance(value, str) and value.strip()]
                    resume_text = max(values, key=len, default="")

                if not resume_text:
                    continue

                output = target_dir / f"kaggle_{csv_file.stem}_{idx:04d}.txt"
                output.write_text(resume_text, encoding="utf-8")
                extracted += 1
    return extracted


def import_kaggle_dataset(target_resume_dir: str = "dataset/resumes") -> None:
    target_dir = Path(target_resume_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
    copied_files = _copy_supported_files(dataset_path, target_dir)
    extracted_from_csv = _extract_resumes_from_csv(dataset_path, target_dir)

    print(f"Downloaded Kaggle dataset to: {dataset_path}")
    print(f"Copied supported files: {copied_files}")
    print(f"Extracted text resumes from CSV rows: {extracted_from_csv}")
    print(f"Total imported resumes: {copied_files + extracted_from_csv}")


if __name__ == "__main__":
    import_kaggle_dataset()

