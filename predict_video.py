import argparse
from dataclasses import dataclass
from pathlib import Path
import cv2
from ultralytics import YOLO
import torch
def get_compatible_device() -> str:
    """Перевіряє сумісність GPU з поточним середовищем PyTorch.

    Якщо Compute Capability < 7.0 (наприклад, GTX 1050 Ti = 6.1) і PyTorch не
    має відповідних ядер — безпечно повертає 'cpu'.
    """
    if not torch.cuda.is_available():
        print("[INFO] CUDA недоступна. Використовуємо CPU.")
        return "cpu"

    try:
        capability = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)

        # Для PyTorch з CUDA 12+ зазвичай потрібна Compute Capability >= 7.0
        if capability[0] < 7:
            print(
                f"[INFO] Виявлено GPU: {gpu_name} (Compute Capability {capability[0]}.{capability[1]})."
            )
            print(
                "[INFO] Поточна збірка PyTorch не має ядер для цієї архітектури. Перемикаємося на CPU."
            )
            return "cpu"

        # Тестова операція розвороту тензора на GPU для виявлення помилок
        test_tensor = torch.zeros(1, device="cuda")
        _ = test_tensor.flip(0)

        print(f"[INFO] Успішно активовано GPU: {gpu_name}")
        return "cuda:0"

    except Exception as error:
        print(
            f"[WARNING] Тестування GPU завершилося помилкою: {error}. Перемикаємося на CPU."
        )
        return "cpu"


class VideoAnnotator:

    def __init__(
        self,
        weights_path: str,
        folder_path: str,
        output_path: str,
        show_preview: bool = True,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.folder_path = Path(folder_path)
        self.output_path = Path(output_path)
        self.show_preview = show_preview

        # Автоматичне визначення безпечного пристрою (GPU або CPU)
        self.device = get_compatible_device()

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Файл із вагами моделі не знайдено: {self.weights_path}"
            )

        print(f"[INFO] Завантаження моделі YOLO з {self.weights_path}...")
        self.model = YOLO(str(self.weights_path))

    def _process_single_video(
        self, video_path: Path, save_path: Path
    ) -> bool:
        """Обробляє один відеофайл та зберігає аннотований результат."""
        print(f"\n[INFO] Обробка відео: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Не вдалося відкрити відеофайл: {video_path}")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(save_path), fourcc, fps, (width, height)
        )

        is_interrupted = False

        # Виклик моделі з передачею сумісного пристрою (cuda:0 або cpu)
        results = self.model.predict(
            source=str(video_path),
            device=self.device,
            stream=True,
            verbose=False,
        )

        for result in results:
            annotated_frame = result.plot()
            out.write(annotated_frame)

            if self.show_preview:
                cv2.imshow("Bird Detector Preview", annotated_frame)
                # Натисніть 'q' для переходу до наступного відео, або Esc для виходу
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] Пропуск поточного відео за запитом.")
                    break
                elif key == 27:  # ESC
                    print("[INFO] Переривання обробки користувачем.")
                    is_interrupted = True
                    break

        cap.release()
        out.release()

        if self.show_preview:
            cv2.destroyAllWindows()

        print(f"[INFO] Збережено результат у: {save_path}")
        return is_interrupted

    def run(self) -> None:
        """Запускає обробку всіх відеофайлів у вказаній директорії."""
        if not self.folder_path.exists():
            raise FileNotFoundError(
                f"Директорію з відео не знайдено: {self.folder_path}"
            )

        self.output_path.mkdir(parents=True, exist_ok=True)

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}
        video_files = [
            p
            for p in self.folder_path.iterdir()
            if p.is_file() and p.suffix in video_extensions
        ]

        if not video_files:
            print(
                f"[WARNING] Не знайдено відеофайлів у папці: {self.folder_path}"
            )
            return

        print(f"[INFO] Знайдено {len(video_files)} відео для обробки.")

        for video_path in sorted(video_files):
            # Пропускаємо вже оброблені файли (з префіксом result_)
            if video_path.stem.startswith("result_"):
                continue

            output_file = (
                self.output_path / f"result_{video_path.stem}.mp4"
            )
            is_interrupted = self._process_single_video(
                video_path, output_file
            )

            if is_interrupted:
                break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Пакетна обробка відео за допомогою YOLO"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="examples",
        help="Шлях до папки з вхідними відео",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="examples",
        help="Шлях до папки для збереження результатів",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Шлях до файлу ваг YOLO (.pt)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Вимкнути відображення вікна попереднього перегляду (для headless/Docker)",
    )

    args = parser.parse_args()

    annotator = VideoAnnotator(
        weights_path=args.weights,
        folder_path=args.folder,
        output_path=args.output,
        show_preview=not args.no_preview,
    )
    annotator.run()


if __name__ == "__main__":
    main()