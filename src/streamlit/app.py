"""
Streamlit приложение для предсказания цен на квартиры в Москве

Приложение загружает обученную модель и позволяет пользователю
вводить параметры квартиры для получения предсказания цены.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import sys
from pathlib import Path

# Добавляем корень проекта в путь для импорта
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import DataPreprocessor

# Настройка страницы
st.set_page_config(
    page_title="Предсказание цен на квартиры в Москве",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Пути к моделям
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

@st.cache_resource
def load_model_and_preprocessor():
    """Загружает обученную модель и препроцессор"""
    try:
        # Загружаем результаты обучения
        results_path = MODELS_DIR / "training_results.json"
        if results_path.exists():
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Находим лучшую модель по R²
            best_model_name = max(results['models'].keys(), 
                                 key=lambda k: results['models'][k]['val']['r2'])
            st.sidebar.success(f"✅ Загружена модель: **{best_model_name}**")
        else:
            # По умолчанию используем CatBoost
            best_model_name = "catboost"
            st.sidebar.warning("⚠️ Файл результатов не найден, используется CatBoost")
        
        # Загружаем модель и препроцессор
        safe_model_name = best_model_name.replace(" ", "_").lower()
        model_path = MODELS_DIR / f"{safe_model_name}.pkl"
        preprocessor_path = MODELS_DIR / "preprocessor.pkl"
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        return model, preprocessor, best_model_name, results if results_path.exists() else None
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None, None, None, None

@st.cache_data
def load_dataset_stats():
    """Загружает статистику по датасету"""
    try:
        data_path = DATA_DIR / "moscow_housing_merged.csv"
        df = pd.read_csv(data_path)
        
        stats = {
            'count': len(df),
            'price_mean': df['price'].mean(),
            'price_median': df['price'].median(),
            'price_std': df['price'].std(),
            'price_min': df['price'].min(),
            'price_max': df['price'].max(),
            'total_area_mean': df['total_area'].mean(),
            'rooms_mean': df['rooms'].mean(),
            'year_range': (df['year'].min(), df['year'].max()),
        }
        return stats, df
    except Exception as e:
        st.warning(f"⚠️ Не удалось загрузить статистику: {e}")
        return None, None

def main():
    """Основная функция приложения"""
    
    # Заголовок
    st.title("🏠 Предсказание цен на квартиры в Москве")
    st.markdown("""
    Данное приложение использует машинное обучение для предсказания цены квартиры 
    на основе её характеристик: площади, количества комнат, этажа и других параметров.
    """)
    
    # Загрузка модели
    model, preprocessor, model_name, results = load_model_and_preprocessor()
    if model is None or preprocessor is None:
        st.error("❌ Не удалось загрузить модель. Убедитесь, что модели обучены.")
        st.info("💡 Запустите `notebooks/training_pipeline.ipynb` для обучения моделей.")
        return
    
    # Загрузка статистики
    stats, df = load_dataset_stats()
    
    # Боковая панель с информацией
    st.sidebar.title("📊 Информация о проекте")
    st.sidebar.markdown("""
    ### О датасете
    Данные объединены из 4 источников Kaggle:
    - Prices of Moscow apartments (2024)
    - Moscow Apartment Listings (2020)
    - Price of flats in Moscow (2018)
    - Moscow Housing Price Dataset
    """)
    
    if stats:
        st.sidebar.markdown(f"""
        ### Статистика датасета
        - **Количество квартир**: {stats['count']:,}
        - **Средняя цена**: {stats['price_mean']:,.0f} ₽
        - **Медианная цена**: {stats['price_median']:,.0f} ₽
        - **Диапазон годов**: {int(stats['year_range'][0])}–{int(stats['year_range'][1])}
        """)
    
    if results:
        st.sidebar.markdown(f"""
        ### Метрики модели ({model_name})
        - **MAE**: {results['models'][model_name]['val']['mae']:,.0f} ₽
        - **RMSE**: {results['models'][model_name]['val']['rmse']:,.0f} ₽
        - **R²**: {results['models'][model_name]['val']['r2']:.4f}
        - **MAPE**: {results['models'][model_name]['val']['mape']:.2f}%
        """)
    
    # Основная панель ввода
    st.header("🔧 Введите параметры квартиры")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_area = st.number_input(
            "Общая площадь (м²)",
            min_value=10.0,
            max_value=500.0,
            value=60.0,
            step=5.0,
            help="Общая площадь квартиры в квадратных метрах"
        )
        
        rooms = st.number_input(
            "Количество комнат",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Количество комнат в квартире"
        )
    
    with col2:
        floor = st.number_input(
            "Этаж",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            help="Этаж, на котором расположена квартира"
        )
        
        total_floors = st.number_input(
            "Этажность дома",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Общее количество этажей в доме"
        )
    
    with col3:
        year = st.number_input(
            "Год данных",
            min_value=1950,
            max_value=2025,
            value=2015,
            step=1,
            help="Год постройки дома или год данных"
        )
        
        subway_dist = st.number_input(
            "Расстояние до метро м",
            min_value=0.0,
            max_value=100000.0,
            value=1.0,
            step=0.1,
            help="Расстояние до ближайшей станции метро в километрах"
        )
    
    # Валидация
    if floor > total_floors:
        st.warning("⚠️ Этаж не может быть больше этажности дома")
        total_floors = floor
    
    # Кнопка предсказания
    if st.button("💰 Предсказать цену", type="primary"):
        # Подготовка данных (порядок колонок должен совпадать с обучением!)
        # Используем feature_names из препроцессора для правильного порядка
        if hasattr(preprocessor, 'feature_names') and preprocessor.feature_names:
            # Создаем данные в правильном порядке
            input_data = pd.DataFrame([[
                total_area, rooms, floor, total_floors, subway_dist, year
            ]], columns=preprocessor.feature_names)
        else:
            # Фоллбэк на стандартный порядок
            input_data = pd.DataFrame({
                'total_area': [total_area],
                'rooms': [rooms],
                'floor': [floor],
                'total_floors': [total_floors],
                'subway_dist': [subway_dist],
                'year': [year]
            })
        
        try:
            # Предобработка
            X_processed = preprocessor.transform(input_data)
            
            # Предсказание
            prediction = model.predict(X_processed)[0]
            
            # Отображение результата
            st.success("✅ Предсказание выполнено!")
            
            # Основной результат
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Предсказанная цена",
                    value=f"{prediction:,.0f} ₽",
                    delta=None
                )
            
            if stats:
                with col2:
                    diff_from_mean = ((prediction - stats['price_mean']) / stats['price_mean']) * 100
                    st.metric(
                        label="Относительно средней цены",
                        value=f"{diff_from_mean:+.1f}%",
                        delta=f"{prediction - stats['price_mean']:,.0f} ₽"
                    )
                
                with col3:
                    price_per_sqm = prediction / total_area
                    st.metric(
                        label="Цена за м²",
                        value=f"{price_per_sqm:,.0f} ₽/м²"
                    )
            
            # Дополнительная информация
            st.markdown("---")
            st.subheader("📊 Анализ предсказания")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Характеристики квартиры")
                st.markdown(f"""
                - **Площадь**: {total_area} м²
                - **Комнат**: {rooms}
                - **Этаж**: {floor} из {total_floors}
                - **Год**: {year}
                - **До метро**: {subway_dist} км
                """)
            
            with col2:
                if stats:
                    st.markdown("#### Сравнение с рынком")
                    percentile = (prediction < df['price']).sum() / len(df) * 100
                    st.markdown(f"""
                    - **Перцентиль**: {percentile:.1f}%
                    - **Средняя цена на рынке**: {stats['price_mean']:,.0f} ₽
                    - **Медианная цена**: {stats['price_median']:,.0f} ₽
                    - **Диапазон цен**: {stats['price_min']:,.0f}–{stats['price_max']:,.0f} ₽
                    """)
                    
                    if prediction < stats['price_mean'] * 0.7:
                        st.info("💡 Квартира значительно дешевле средней по рынку")
                    elif prediction > stats['price_mean'] * 1.3:
                        st.info("💡 Квартира значительно дороже средней по рынку")
                    else:
                        st.info("💡 Цена соответствует среднерыночной")
        
        except Exception as e:
            st.error(f"❌ Ошибка при предсказании: {e}")
    
    # Футер
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><b>Проект по курсу "Введение в машинное обучение"</b></p>
        <p>Авторы: Деружинский Дмитрий, Токарев Алексей | 2024-2025</p>
        <p><a href="https://github.com/your-repo" target="_blank">GitHub</a> | 
        <a href="https://www.kaggle.com" target="_blank">Kaggle Datasets</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
