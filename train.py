from dataclasses import dataclass
from pathlib import Path
import random
import shutil
import gdown
from ultralytics import YOLO


@dataclass
class DatasetConfig:
   
    gdrive_folder_id: str = "14c0cV2_apCLsbDr3V9HXju-BDCaPMAwE"


    download_dir: Path = Path("train_data")


    images_dir: Path = Path("train_data/bird/images")
    labels_dir: Path = Path("train_data/bird/labels")
    output_dir: Path = Path("dataset")

   
    train_ratio: float = 0.8
    seed: int = 42
    source_class_id: int = 24
    target_class_id: int = 0
    class_name: str = "bird"


def download_uncompressed_folder(config: DatasetConfig) -> None:
    """Завантажує неархівовану папку з Google Drive."""
    if not config.images_dir.exists():
        print("=" * 50)
        print("Downloading uncompressed dataset folder from Google Drive...")
        print("=" * 50)

        url = f"https://drive.google.com/drive/folders/{config.gdrive_folder_id}"

        gdown.download_folder(
            url=url,
            output=str(config.download_dir),
            quiet=False,
            use_cookies=False,
        )
    else:
        print("Dataset directory already exists. Skipping download.")


def create_directories(config: DatasetConfig) -> None:
    """Створює структуру директорій для обробленого датасету YOLO."""
    directories = [
        config.output_dir / "images" / "train",
        config.output_dir / "images" / "val",
        config.output_dir / "labels" / "train",
        config.output_dir / "labels" / "val",
        Path("weights"),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_images(config: DatasetConfig) -> list[Path]:
    """Знаходить усі зображення у завантаженій папці."""
    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [
        path
        for path in config.images_dir.iterdir()
        if path.is_file() and path.suffix in extensions
    ]
    return sorted(images)


def validate_label(label_path: Path, config: DatasetConfig) -> list[str]:
    """Конвертує ID класу (24 -> 0) та перевіряє коректність координат."""
    converted_lines = []
    with label_path.open("r") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation in {label_path}, line {line_number}: {line}"
                )

            class_id = int(parts[0])
            if class_id != config.source_class_id:
                raise ValueError(
                    f"Unexpected class ID {class_id} in {label_path}. Expected: {config.source_class_id}"
                )

            coordinates = [float(p) for p in parts[1:]]
            for value in coordinates:
                if not 0 <= value <= 1:
                    raise ValueError(
                        f"Invalid YOLO coordinate in {label_path}, line {line_number}: {value}"
                    )

            parts[0] = str(config.target_class_id)
            converted_lines.append(" ".join(parts))

    return converted_lines


def process_image(image_path: Path, split: str, config: DatasetConfig) -> None:
    """Копіює зображення та створює файл розмітки."""
    label_path = config.labels_dir / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(
            f"Warning: Label file not found for {image_path.name}. Creating empty label (background sample)."
        )
        converted_lines = []
    else:
        converted_lines = validate_label(label_path, config)

    output_image = config.output_dir / "images" / split / image_path.name
    output_label = (
        config.output_dir / "labels" / split / f"{image_path.stem}.txt"
    )

    shutil.copy2(image_path, output_image)

    with output_label.open("w") as file:
        file.write("\n".join(converted_lines))


def create_data_yaml(config: DatasetConfig) -> None:
    """Генерує файл data.yaml без лишніх відступів."""
    yaml_content = f"""path: {config.output_dir.resolve()}
train: images/train
val: images/val
names:
  0: {config.class_name}
"""
    yaml_path = config.output_dir / "data.yaml"
    with yaml_path.open("w") as file:
        file.write(yaml_content)

    print(f"Created: {yaml_path}")


def train_yolo(data_yaml: Path) -> Path:
    """Запускає процес навчання YOLO."""
    print("\n" + "=" * 50)
    print("Starting Model Training")
    print("=" * 50)

    base_model = "yolo11n.pt" if Path("yolo11n.pt").exists() else "yolov26n.pt"
    model = YOLO(base_model)

    results = model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=640,
        batch=16,
        project="runs/detect",
        name="bird_model",
        exist_ok=True,
    )

    best_weights_src = Path(results.save_dir) / "weights" / "best.pt"
    target_weights_dst = Path("weights/best.pt")

    if best_weights_src.exists():
        shutil.copy2(best_weights_src, target_weights_dst)
        print(f"\nSaved best model weights to: {target_weights_dst.resolve()}")

    return target_weights_dst


def main() -> None:
    config = DatasetConfig()

    download_uncompressed_folder(config)

    if not config.images_dir.exists():
        raise FileNotFoundError(
            f"Source images directory not found: {config.images_dir}"
        )

    create_directories(config)
    images = get_images(config)

    if not images:
        raise RuntimeError(f"No images found in: {config.images_dir}")

    print("=" * 50)
    print("Bird Dataset Preparation (with Background Support)")
    print("=" * 50)

    random.seed(config.seed)
    random.shuffle(images)

    split_index = int(len(images) * config.train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    print(f"Processing {len(train_images)} training samples...")
    for image_path in train_images:
        process_image(image_path, "train", config)

    print(f"Processing {len(val_images)} validation samples...")
    for image_path in val_images:
        process_image(image_path, "val", config)

    create_data_yaml(config)

    data_yaml_path = config.output_dir / "data.yaml"
    train_yolo(data_yaml_path)


if __name__ == "__main__":
    main()
