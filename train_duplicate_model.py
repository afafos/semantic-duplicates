#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обучения модели определения семантических дубликатов новостей
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import joblib
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DuplicateNewsModel:
    """Модель для определения семантических дубликатов новостей"""
    
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        """
        Инициализация модели
        
        Args:
            model_name: название модели для эмбеддингов из sentence-transformers
        """
        print(f"Загрузка модели эмбеддингов: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        self.classifier = None
        self.feature_names = []
        
    def extract_features(self, text1, text2):
        """
        Извлечение признаков из пары текстов
        
        Args:
            text1: первый текст
            text2: второй текст
            
        Returns:
            numpy array с признаками
        """
        # Получаем эмбеддинги
        emb1 = self.embedding_model.encode([text1])[0]
        emb2 = self.embedding_model.encode([text2])[0]
        
        # Косинусное сходство
        cosine_sim = cosine_similarity([emb1], [emb2])[0][0]
        
        # Евклидово расстояние
        euclidean_dist = np.linalg.norm(emb1 - emb2)
        
        # Манхэттенское расстояние
        manhattan_dist = np.sum(np.abs(emb1 - emb2))
        
        # Длины текстов
        len1 = len(text1)
        len2 = len(text2)
        len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        len_diff = abs(len1 - len2)
        
        # Количество слов
        words1 = len(text1.split())
        words2 = len(text2.split())
        word_ratio = min(words1, words2) / max(words1, words2) if max(words1, words2) > 0 else 0
        word_diff = abs(words1 - words2)
        
        # Пересечение слов
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        jaccard = len(set1.intersection(set2)) / len(set1.union(set2)) if len(set1.union(set2)) > 0 else 0
        
        # Простые текстовые метрики
        common_chars = sum((text1.lower().count(c) + text2.lower().count(c)) / 2 
                          for c in set(text1.lower() + text2.lower()))
        
        features = np.array([
            cosine_sim,
            euclidean_dist,
            manhattan_dist,
            len_ratio,
            len_diff,
            word_ratio,
            word_diff,
            jaccard,
            common_chars
        ])
        
        return features
    
    def prepare_features(self, df, text_col1='text1', text_col2='text2', batch_size=32):
        """
        Подготовка признаков для всего датасета
        
        Args:
            df: pandas DataFrame с данными
            text_col1: название колонки с первым текстом
            text_col2: название колонки с вторым текстом
            batch_size: размер батча для обработки
            
        Returns:
            numpy array с признаками для всех примеров
        """
        print("Извлечение признаков из данных...")
        features_list = []
        
        for idx in tqdm(range(len(df)), desc="Обработка"):
            text1 = str(df.iloc[idx][text_col1])
            text2 = str(df.iloc[idx][text_col2])
            features = self.extract_features(text1, text2)
            features_list.append(features)
        
        self.feature_names = [
            'cosine_similarity',
            'euclidean_distance',
            'manhattan_distance',
            'length_ratio',
            'length_diff',
            'word_ratio',
            'word_diff',
            'jaccard_similarity',
            'common_chars'
        ]
        
        return np.array(features_list)
    
    def train(self, X_train, y_train, classifier_type='xgboost'):
        """
        Обучение классификатора
        
        Args:
            X_train: признаки для обучения
            y_train: метки для обучения
            classifier_type: тип классификатора ('logistic', 'random_forest', 'gradient_boosting', 'xgboost')
        """
        print(f"Обучение классификатора: {classifier_type}...")
        
        if classifier_type == 'logistic':
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif classifier_type == 'gradient_boosting':
            self.classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif classifier_type == 'xgboost':
            self.classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Неизвестный тип классификатора: {classifier_type}")
        
        self.classifier.fit(X_train, y_train)
        print("✅ Обучение завершено!")
    
    def predict(self, text1, text2):
        """
        Предсказание для пары текстов
        
        Args:
            text1: первый текст
            text2: второй текст
            
        Returns:
            tuple (prediction, probability)
        """
        if self.classifier is None:
            raise ValueError("Модель не обучена! Вызовите train() сначала.")
        
        features = self.extract_features(text1, text2)
        features = features.reshape(1, -1)
        
        prediction = self.classifier.predict(features)[0]
        probability = self.classifier.predict_proba(features)[0]
        
        return prediction, probability
    
    def evaluate(self, X_test, y_test):
        """
        Оценка модели на тестовых данных
        
        Args:
            X_test: признаки для тестирования
            y_test: метки для тестирования
            
        Returns:
            dict с метриками
        """
        print("\nОценка модели на тестовых данных...")
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        print("\n" + "="*60)
        print("МЕТРИКИ МОДЕЛИ")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        
        print("\n" + "="*60)
        print("МАТРИЦА ОШИБОК")
        print("="*60)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print("\n" + "="*60)
        print("ПОДРОБНЫЙ ОТЧЕТ")
        print("="*60)
        print(classification_report(y_test, y_pred, target_names=['Не дубликаты', 'Дубликаты']))
        
        # Важность признаков (если доступно)
        if hasattr(self.classifier, 'feature_importances_'):
            print("\n" + "="*60)
            print("ВАЖНОСТЬ ПРИЗНАКОВ")
            print("="*60)
            importances = self.classifier.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i, idx in enumerate(indices):
                print(f"{i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        return metrics, y_pred, y_pred_proba
    
    def save(self, model_path):
        """
        Сохранение модели
        
        Args:
            model_path: путь для сохранения модели
        """
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'classifier': self.classifier,
            'feature_names': self.feature_names,
            'embedding_model_name': self.embedding_model.__class__.__name__
        }
        
        joblib.dump(model_data, model_path)
        print(f"✅ Модель сохранена: {model_path}")
    
    def load(self, model_path):
        """
        Загрузка модели
        
        Args:
            model_path: путь к сохраненной модели
        """
        model_data = joblib.load(model_path)
        self.classifier = model_data['classifier']
        self.feature_names = model_data['feature_names']
        print(f"✅ Модель загружена: {model_path}")


def main():
    """Основная функция"""
    
    # Параметры
    DATA_PATH = Path(__file__).parent / "unified_news_pairs.csv"
    MODEL_DIR = Path(__file__).parent / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    
    print("="*60)
    print("ОБУЧЕНИЕ МОДЕЛИ ОПРЕДЕЛЕНИЯ ДУБЛИКАТОВ НОВОСТЕЙ")
    print("="*60)
    
    # 1. Загрузка данных
    print(f"\n1. Загрузка данных из {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Загружено записей: {len(df)}")
    print(f"   Колонки: {list(df.columns)}")
    print(f"\nРаспределение меток:")
    print(df['label'].value_counts())
    print(f"   Дубликатов: {df['label'].sum()} ({df['label'].sum()/len(df)*100:.2f}%)")
    print(f"   Не дубликатов: {(1-df['label']).sum()} ({(1-df['label']).sum()/len(df)*100:.2f}%)")
    
    # Для тестирования возьмем подмножество (можно убрать для полного обучения)
    # if len(df) > 10000:
    #    print(f"\n⚠️  Для ускорения используется подмножество из 10000 записей")
    #    df = df.sample(n=10000, random_state=RANDOM_STATE)
    
    # 2. Инициализация модели
    print("\n2. Инициализация модели...")
    model = DuplicateNewsModel()
    
    # 3. Извлечение признаков
    print("\n3. Извлечение признаков...")
    X = model.prepare_features(df)
    y = df['label'].values
    
    # 4. Обучение моделей на ВСЕХ данных
    print("\n4. Обучение моделей на полном датасете...")
    print(f"   Всего примеров для обучения: {len(X)}")
    print(f"   Дубликатов: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"   Не дубликатов: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.1f}%)")
    
    classifiers = ['logistic', 'random_forest', 'gradient_boosting', 'xgboost']
    
    for clf_type in classifiers:
        print("\n" + "="*60)
        print(f"ОБУЧЕНИЕ: {clf_type.upper()}")
        print("="*60)
        
        # Создание и обучение модели
        model_instance = DuplicateNewsModel()
        model_instance.feature_names = model.feature_names
        model_instance.train(X, y, classifier_type=clf_type)
        
        # Сохранение
        model_path = MODEL_DIR / f"duplicate_model_{clf_type}.joblib"
        model_instance.save(model_path)
        print(f"✅ Модель сохранена: {model_path.name}")
    
    # 5. Итог
    print("\n" + "="*60)
    print("✅ ВСЕ МОДЕЛИ ОБУЧЕНЫ И СОХРАНЕНЫ")
    print("="*60)
    print(f"\n📁 Сохранено {len(classifiers)} моделей в {MODEL_DIR}")
    print("\n💡 Для валидации моделей запустите:")
    print("   python validate_models.py")
    
    # 6. Быстрое тестирование на примерах (проверка работоспособности)
    print("\n" + "="*60)
    print("БЫСТРАЯ ПРОВЕРКА РАБОТОСПОСОБНОСТИ")
    print("="*60)
    
    # Загружаем XGBoost модель для проверки
    test_model = DuplicateNewsModel()
    test_model.load(MODEL_DIR / "duplicate_model_xgboost.joblib")
    
    # Примеры для тестирования
    test_examples = [
        ("Президент подписал новый закон о налогах.", 
         "Глава государства утвердил налоговый закон.", 
         True),
        ("Завтра будет дождь.", 
         "В России открылся новый завод по производству автомобилей.", 
         False),
        ("Компания Apple представила новый iPhone.", 
         "Apple анонсировала последнюю модель iPhone.", 
         True),
    ]
    
    print("\nТестирование XGBoost модели на примерах:")
    for i, (text1, text2, expected) in enumerate(test_examples, 1):
        prediction, probability = test_model.predict(text1, text2)
        status = '✅' if (prediction == 1) == expected else '❌'
        result = 'Дубликат' if prediction == 1 else 'Не дубликат'
        print(f"  {i}. {status} Предсказание: {result} (вероятность: {probability[1]:.2%})")
    
    print("\n" + "="*60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*60)
    print("\n📊 Для полной валидации на размеченных данных запустите:")
    print("   python validate_models.py")


if __name__ == "__main__":
    main()

