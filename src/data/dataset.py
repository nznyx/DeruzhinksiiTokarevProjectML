"""
Загрузчик датасета для обучения моделей предсказания цен на квартиры в Москве.
Загружает обработанный объединенный датасет и предоставляет методы для ML workflows.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MoscowHousingDataset:
    """
    Класс для загрузки датасета цен на квартиры в Москве.
    
    Возможности:
    - Загрузка предобработанных CSV данных
    - Обработка пропущенных значений
    - Разделение данных на train/val/test
    - Нормализация признаков
    - Возврат данных в форматах pandas и numpy
    """
    
    def __init__(
        self,
        data_path: str = "data/processed/moscow_housing_merged.csv",
        target_column: str = "price",
        drop_columns: Optional[List[str]] = None,
        handle_missing: str = "drop",  # 'drop', 'mean', 'median'
        random_state: int = 42
    ):
        """
        Инициализация загрузчика датасета.
        
        Параметры:
            data_path: Путь к предобработанному CSV файлу
            target_column: Имя целевой колонки (по умолчанию: 'price')
            drop_columns: Список колонок для удаления из признаков (по умолчанию: None)
            handle_missing: Стратегия обработки пропусков ('drop', 'mean', 'median')
            random_state: Seed для воспроизводимости результатов
        """
        self.data_path = data_path
        self.target_column = target_column
        self.drop_columns = drop_columns or []
        self.handle_missing = handle_missing
        self.random_state = random_state
        
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        self.feature_names = None
        
        self._load_data()
    
    def _load_data(self):
        """Загрузить предобработанный датасет."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Файл данных не найден: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Загружен датасет: {self.df.shape[0]} строк, {self.df.shape[1]} колонок")
        logger.info(f"Колонки: {list(self.df.columns)}")
        
        # Вывод информации о пропусках
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            logger.info("Пропущенные значения:")
            for col, count in missing[missing > 0].items():
                logger.info(f"  {col}: {count} ({count/len(self.df)*100:.2f}%)")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработать пропущенные значения согласно выбранной стратегии."""
        if self.handle_missing == "drop":
            before = len(df)
            df = df.dropna()
            after = len(df)
            if before != after:
                logger.info(f"Удалено {before - after} строк с пропущенными значениями")
        
        elif self.handle_missing == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    logger.info(f"Заполнены пропуски в '{col}' средним значением: {mean_val:.2f}")
        
        elif self.handle_missing == "median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Заполнены пропуски в '{col}' медианой: {median_val:.2f}")
        
        return df
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Подготовить признаки и целевую переменную.
        
        Возвращает:
            X: DataFrame с признаками
            y: Series с целевой переменной
        """
        df = self.df.copy()
        
        # Обработка пропусков
        df = self._handle_missing_values(df)
        
        # Отделение целевой переменной
        if self.target_column not in df.columns:
            raise ValueError(f"Целевая колонка '{self.target_column}' не найдена в датасете")
        
        y = df[self.target_column]
        
        # Подготовка признаков
        feature_columns = [col for col in df.columns if col != self.target_column]
        feature_columns = [col for col in feature_columns if col not in self.drop_columns]
        
        X = df[feature_columns]
        self.feature_names = feature_columns
        
        logger.info(f"Признаки подготовлены: {len(feature_columns)} колонок")
        logger.info(f"Названия признаков: {feature_columns}")
        logger.info(f"Размер датасета: {len(X)} образцов")
        
        return X, y
    
    def split_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        shuffle: bool = True
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Разделить данные на train, validation и test выборки.
        
        Параметры:
            test_size: Доля данных для test выборки (по умолчанию: 0.2)
            val_size: Доля оставшихся данных для validation выборки (по умолчанию: 0.1)
            shuffle: Перемешивать ли данные перед разделением (по умолчанию: True)
        
        Возвращает:
            Словарь с ключами 'train', 'val', 'test', содержащими кортежи (X, y)
        """
        X, y = self.prepare_features()
        
        # Первое разделение: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            shuffle=shuffle
        )
        
        # Второе разделение: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            shuffle=shuffle
        )
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        logger.info(f"Данные разделены:")
        logger.info(f"  Train: {len(X_train)} образцов ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"  Val:   {len(X_val)} образцов ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"  Test:  {len(X_test)} образцов ({len(X_test)/len(X)*100:.1f}%)")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def scale_features(
        self,
        scaler_type: str = "standard"
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Нормализовать признаки с использованием StandardScaler или других скейлеров.
        Должно быть вызвано после split_data().
        
        Параметры:
            scaler_type: Тип скейлера ('standard', 'minmax', 'robust')
        
        Возвращает:
            Словарь с нормализованными train, val, test выборками в виде numpy массивов
        """
        if self.X_train is None:
            raise ValueError("Необходимо вызвать split_data() перед нормализацией")
        
        if scaler_type == "standard":
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Неизвестный тип скейлера: {scaler_type}")
        
        # Обучение на train, трансформация всех выборок
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_val_scaled = self.scaler.transform(self.X_val)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info(f"Признаки нормализованы с использованием {scaler_type} скейлера")
        
        return {
            'train': (X_train_scaled, self.y_train.values),
            'val': (X_val_scaled, self.y_val.values),
            'test': (X_test_scaled, self.y_test.values)
        }
    
    def get_data_info(self) -> Dict:
        """Получить информацию о датасете."""
        if self.df is None:
            return {}
        
        info = {
            'total_samples': len(self.df),
            'n_features': len(self.df.columns) - 1,
            'target_column': self.target_column,
            'feature_names': list(self.df.columns.drop(self.target_column)),
            'missing_values': self.df.isnull().sum().to_dict(),
            'target_stats': {
                'mean': self.df[self.target_column].mean(),
                'median': self.df[self.target_column].median(),
                'std': self.df[self.target_column].std(),
                'min': self.df[self.target_column].min(),
                'max': self.df[self.target_column].max()
            }
        }
        
        if self.X_train is not None:
            info['train_size'] = len(self.X_train)
            info['val_size'] = len(self.X_val)
            info['test_size'] = len(self.X_test)
        
        return info
    
    def __repr__(self):
        if self.df is None:
            return "MoscowHousingDataset(не загружен)"
        
        info = f"MoscowHousingDataset(\n"
        info += f"  образцов={len(self.df)},\n"
        info += f"  признаков={len(self.df.columns)-1},\n"
        info += f"  target='{self.target_column}'"
        
        if self.X_train is not None:
            info += f",\n  train={len(self.X_train)}, val={len(self.X_val)}, test={len(self.X_test)}"
        
        info += "\n)"
        return info
