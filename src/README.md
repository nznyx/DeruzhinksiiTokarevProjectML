# Source Code

Исходный код проекта по предсказанию цен на квартиры в Москве.

## Структура

### `data/`
Модули для работы с данными:
- **`loader.py`**: Класс `MoscowHousingLoader` для загрузки и объединения датасетов из raw источников
- **`schemas.py`**: Конфигурация маппингов колонок и унифицированной схемы
- **`prepare_dataset.py`**: Скрипт для подготовки объединенного датасета
- **`dataset.py`**: Класс `MoscowHousingDataset` для загрузки обработанных данных и подготовки к обучению

### `streamlit/`
Интерактивное веб-приложение на Streamlit для демонстрации модели.
Предназначено для развертывания на Hugging Face Spaces.

## Использование модулей data

### Подготовка данных из raw источников

```python
from src.data.loader import MoscowHousingLoader

# Загрузка и объединение данных из raw датасетов
loader = MoscowHousingLoader(data_dir="data/raw")
df = loader.load_all()

# Сохранение результата
df.to_csv("data/processed/moscow_housing_merged.csv", index=False)
```

### Загрузка обработанных данных для обучения

```python
from src.data.dataset import MoscowHousingDataset

# Инициализация датасета
dataset = MoscowHousingDataset(
    data_path="data/processed/moscow_housing_merged.csv",
    handle_missing="mean",
    random_state=42
)

# Разделение на train/val/test
splits = dataset.split_data(test_size=0.2, val_size=0.1)

# Нормализация признаков
scaled_data = dataset.scale_features(scaler_type="standard")

# Получение данных для обучения
X_train, y_train = scaled_data['train']
X_val, y_val = scaled_data['val']
X_test, y_test = scaled_data['test']
```

## Разработка

При добавлении новых источников данных:
1. Обновите `schemas.py` с маппингами колонок
2. При необходимости расширьте `loader.py`
3. Обновите документацию
