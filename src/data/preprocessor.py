"""
Модуль предобработки данных для обучения моделей машинного обучения.

Содержит класс DataPreprocessor для масштабирования признаков.
"""

import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class DataPreprocessor:
    """
    Класс для предобработки данных перед обучением модели.
    
    Поддерживает различные типы масштабирования:
    - standard: StandardScaler (z-score нормализация)
    - minmax: MinMaxScaler (масштабирование в диапазон [0, 1])
    - robust: RobustScaler (устойчив к выбросам)
    
    Attributes:
        scaler_type (str): Тип используемого скейлера
        scaler: Объект скейлера из sklearn
    
    Example:
        >>> preprocessor = DataPreprocessor(scaler_type='standard')
        >>> X_train_scaled = preprocessor.fit_transform(X_train)
        >>> X_test_scaled = preprocessor.transform(X_test)
    """
    
    def __init__(self, scaler_type='standard'):
        """
        Инициализация препроцессора.
        
        Args:
            scaler_type (str): Тип скейлера ('standard', 'minmax', 'robust').
                По умолчанию 'standard'.
        
        Raises:
            ValueError: Если указан неизвестный тип скейлера.
        """
        valid_types = ['standard', 'minmax', 'robust']
        if scaler_type not in valid_types:
            raise ValueError(
                f"Unknown scaler type: {scaler_type}. "
                f"Valid types: {', '.join(valid_types)}"
            )
        
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_names = None
        
    def fit(self, X_train):
        """
        Обучить скейлер на обучающей выборке.
        
        Args:
            X_train (array-like): Обучающие данные для фитирования скейлера
        
        Returns:
            self: Возвращает self для цепочки вызовов
        
        Raises:
            ValueError: Если тип скейлера некорректен
        """
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        self.scaler.fit(X_train)
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        return self
    
    def transform(self, X):
        """
        Применить преобразование к данным.
        
        Args:
            X (array-like): Данные для преобразования
        
        Returns:
            array-like: Преобразованные данные
        
        Raises:
            ValueError: Если скейлер еще не обучен (не вызван fit)
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        """
        Обучить скейлер и применить преобразование.
        
        Args:
            X (array-like): Данные для обучения и преобразования
        
        Returns:
            array-like: Преобразованные данные
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Обратное преобразование (денормализация).
        
        Args:
            X_scaled (array-like): Масштабированные данные
        
        Returns:
            array-like: Исходные данные
        
        Raises:
            ValueError: Если скейлер еще не обучен
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.inverse_transform(X_scaled)
    
    def save(self, path):
        """
        Сохранить препроцессор в файл.
        
        Args:
            path (str or Path): Путь для сохранения файла
        """
        joblib.dump(self, path)
        print(f"✅ Препроцессор сохранен: {path}")
    
    @staticmethod
    def load(path):
        """
        Загрузить препроцессор из файла.
        
        Args:
            path (str or Path): Путь к файлу препроцессора
        
        Returns:
            DataPreprocessor: Загруженный препроцессор
        """
        preprocessor = joblib.load(path)
        print(f"✅ Препроцессор загружен: {path}")
        return preprocessor

