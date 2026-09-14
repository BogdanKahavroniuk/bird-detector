# інстркуція

Швидкий запуск (Docker)

1. **Розмістіть дані:**
   - Зображення та `.txt` розмітку: `train_data/bird/images/` та `train_data/bird/labels/`
   - Вхідні відео: `examples/`

2. **Запустіть навчання: та маркування **
   ```bash
   docker compose up --build train
   docker compose up --build predict_batch

Локальний запуск (без Docker)

    Активація venv та встановлення залежностей:
    Bash

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    Навчання:
    Bash

    python train.py

    Інференс відео:
    Bash
    python predict_video.py --folder examples --output examples --weights weights/best.pt
