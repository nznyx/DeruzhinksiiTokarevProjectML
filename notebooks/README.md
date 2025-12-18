# Notebooks

Эта директория содержит Jupyter ноутбуки для анализа данных и обучения моделей.

## Доступные ноутбуки

### `training_pipeline.ipynb`

Основной ноутбук для обучения моделей предсказания цен на квартиры.

**Содержание:**
- 0. Импорты и настройка окружения (фиксирование random seed)
- 1. Анализ датасета (статистика, визуализация, корреляции)
- 2. Удаление выбросов (метод IQR)
- 3. Разбиение на train/test/val и пайплайн предобработки
- 4. Пайплайн обучения (с валидацией и сохранением лучшего чекпоинта)
- 5. Пайплайн анализа результатов (метрики, residual plots)
- 6. Обучение моделей:
  - Linear Regression (baseline)
  - Random Forest (ансамбль деревьев)
  - Gradient Boosting (sklearn)
  - XGBoost (экстремальный gradient boosting)
  - LightGBM (быстрый gradient boosting)
  - CatBoost (gradient boosting от Yandex)
  - SVR (Support Vector Regression)
- 7. Оценка результатов на тестовой выборке
- 8. Выбор лучшей модели (сравнительный анализ)

**Результаты:**
- Обученные модели сохраняются в `../models/`
- JSON с метриками: `../models/training_results.json`
- Препроцессор: `../models/preprocessor.pkl`

## Использование

Перед началом работы убедитесь, что:
1. Настроено виртуальное окружение (`./setup_venv.sh`)
2. Загружены данные (`./download_data.sh`)
3. Подготовлен датасет (`./prepare_dataset.sh`)

Объединенный датасет находится в `../data/processed/moscow_housing_merged.csv`.

## Запуск Jupyter

```bash
source venv/bin/activate
jupyter notebook
```

Затем откройте `training_pipeline.ipynb` и запустите все ячейки последовательно.
