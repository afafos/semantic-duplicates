#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для валидации обученных моделей на согласованных разметках
Загружает данные из agreed_annotations_osokina_popov.json и проверяет точность предсказаний
"""

import json
from pathlib import Path
from train_duplicate_model import DuplicateNewsModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm


def parse_text_pair(text):
    """
    Парсит текст с двумя новостями, разделенными маркерами
    
    Args:
        text: строка с двумя текстами
        
    Returns:
        tuple (text1, text2)
    """
    parts = text.split("----------ТЕКСТ 2----------")
    
    if len(parts) != 2:
        return None, None
    
    text1 = parts[0].replace("----------ТЕКСТ 1----------", "").strip()
    text2 = parts[1].strip()
    
    return text1, text2


def load_annotations(json_path):
    """
    Загружает согласованные аннотации
    
    Args:
        json_path: путь к JSON файлу
        
    Returns:
        list из словарей с парами текстов и метками
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    parsed_data = []
    
    for item in data:
        text1, text2 = parse_text_pair(item['text'])
        
        if text1 is None or text2 is None:
            continue
        
        # Преобразуем метку в бинарную (0 или 1)
        label = 1 if item['agreed_label'] == 'are_duplicates' else 0
        
        parsed_data.append({
            'id': item['id'],
            'text1': text1,
            'text2': text2,
            'label': label,
            'annotators': item['annotators']
        })
    
    return parsed_data


def evaluate_model_on_annotations(model, data, sample_size=None):
    """
    Оценивает модель на размеченных данных
    
    Args:
        model: обученная модель DuplicateNewsModel
        data: список с размеченными парами текстов
        sample_size: количество примеров для тестирования (None = все)
        
    Returns:
        dict с метриками и предсказаниями
    """
    if sample_size:
        data = data[:sample_size]
    
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    print(f"Оценка на {len(data)} примерах...")
    
    for item in tqdm(data, desc="Предсказание"):
        try:
            prediction, probability = model.predict(item['text1'], item['text2'])
            
            y_true.append(item['label'])
            y_pred.append(prediction)
            y_pred_proba.append(probability[1])  # Вероятность дубликата
            
        except Exception as e:
            print(f"\n⚠️  Ошибка при предсказании для ID {item['id']}: {e}")
            continue
    
    # Вычисляем метрики
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics


def print_model_metrics(model_name, metrics):
    """
    Красиво выводит метрики модели
    
    Args:
        model_name: название модели
        metrics: словарь с метриками
    """
    print(f"\n{'='*80}")
    print(f"МОДЕЛЬ: {model_name}")
    print(f"{'='*80}")
    
    print(f"\n📊 Основные метрики:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    
    print(f"\n📈 Матрица ошибок:")
    cm = metrics['confusion_matrix']
    print(f"                   Predicted")
    print(f"                   Not Dup    Duplicate")
    print(f"   Actual Not Dup    {cm[0][0]:<8}   {cm[0][1]:<8}")
    print(f"   Actual Duplicate  {cm[1][0]:<8}   {cm[1][1]:<8}")
    
    # Вычисляем количество ошибок
    false_positives = cm[0][1]
    false_negatives = cm[1][0]
    total_errors = false_positives + false_negatives
    total_samples = len(metrics['y_true'])
    
    print(f"\n❌ Ошибки:")
    print(f"   False Positives (ложные дубликаты):  {false_positives} ({false_positives/total_samples*100:.2f}%)")
    print(f"   False Negatives (пропущенные дубл.): {false_negatives} ({false_negatives/total_samples*100:.2f}%)")
    print(f"   Всего ошибок: {total_errors} из {total_samples} ({total_errors/total_samples*100:.2f}%)")


def main():
    """Основная функция"""
    
    print("="*80)
    print("ВАЛИДАЦИЯ МОДЕЛЕЙ НА СОГЛАСОВАННЫХ РАЗМЕТКАХ")
    print("="*80)
    
    # Пути к файлам
    annotations_path = Path(__file__).parent / "agreed_annotations_osokina_popov.json"
    models_dir = Path(__file__).parent / "models"
    
    # Проверка наличия файлов
    if not annotations_path.exists():
        print(f"\n❌ Ошибка: Файл с аннотациями не найден: {annotations_path}")
        return
    
    if not models_dir.exists():
        print(f"\n❌ Ошибка: Папка с моделями не найдена: {models_dir}")
        print("   Сначала обучите модели: python train_duplicate_model.py")
        return
    
    # 1. Загрузка данных
    print(f"\n1. Загрузка аннотаций из {annotations_path.name}...")
    data = load_annotations(annotations_path)
    print(f"✅ Загружено примеров: {len(data)}")
    
    # Статистика по меткам
    duplicates_count = sum(1 for item in data if item['label'] == 1)
    not_duplicates_count = len(data) - duplicates_count
    print(f"\nРаспределение:")
    print(f"   Дубликаты:     {duplicates_count} ({duplicates_count/len(data)*100:.1f}%)")
    print(f"   Не дубликаты:  {not_duplicates_count} ({not_duplicates_count/len(data)*100:.1f}%)")
    
    # Опция: использовать подмножество для быстрого тестирования
    SAMPLE_SIZE = None  # Установите число (например, 100) для быстрого теста
    
    if SAMPLE_SIZE and len(data) > SAMPLE_SIZE:
        print(f"\n⚠️  Для ускорения используется подмножество: {SAMPLE_SIZE} примеров")
        data = data[:SAMPLE_SIZE]
    
    # 2. Поиск моделей
    print(f"\n2. Поиск обученных моделей в {models_dir}...")
    model_files = list(models_dir.glob("duplicate_model_*.joblib"))
    
    if not model_files:
        print(f"❌ Ошибка: Не найдено обученных моделей")
        return
    
    print(f"✅ Найдено моделей: {len(model_files)}")
    for mf in sorted(model_files):
        print(f"   - {mf.name}")
    
    # 3. Оценка каждой модели
    print(f"\n{'='*80}")
    print("3. ОЦЕНКА МОДЕЛЕЙ")
    print(f"{'='*80}")
    
    all_results = {}
    
    for model_file in sorted(model_files):
        model_name = model_file.stem.replace("duplicate_model_", "").upper()
        
        try:
            # Загрузка модели
            print(f"\n{'='*80}")
            print(f"Загрузка модели: {model_name}")
            print(f"{'='*80}")
            
            model = DuplicateNewsModel()
            model.load(model_file)
            
            # Оценка
            metrics = evaluate_model_on_annotations(model, data, SAMPLE_SIZE)
            all_results[model_name] = metrics
            
            # Вывод результатов
            print_model_metrics(model_name, metrics)
            
        except Exception as e:
            print(f"\n❌ Ошибка при оценке модели {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. Сравнение моделей
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("4. СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
        print(f"{'='*80}")
        
        print(f"\n{'Модель':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*80)
        
        for model_name, metrics in sorted(all_results.items()):
            print(f"{model_name:<25} "
                  f"{metrics['accuracy']:.4f}       "
                  f"{metrics['precision']:.4f}       "
                  f"{metrics['recall']:.4f}       "
                  f"{metrics['f1']:.4f}")
        
        # Лучшая модель
        best_model = max(all_results.items(), key=lambda x: x[1]['f1'])
        print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ (по F1-Score): {best_model[0]}")
        print(f"   F1-Score: {best_model[1]['f1']:.4f} ({best_model[1]['f1']*100:.2f}%)")
        print(f"   Accuracy: {best_model[1]['accuracy']:.4f} ({best_model[1]['accuracy']*100:.2f}%)")
    
    # 5. Подробный отчет для лучшей модели
    if all_results:
        best_model_name, best_metrics = max(all_results.items(), key=lambda x: x[1]['f1'])
        
        print(f"\n{'='*80}")
        print(f"5. ПОДРОБНЫЙ ОТЧЕТ ДЛЯ ЛУЧШЕЙ МОДЕЛИ: {best_model_name}")
        print(f"{'='*80}")
        
        print("\nClassification Report:")
        print(classification_report(
            best_metrics['y_true'], 
            best_metrics['y_pred'],
            target_names=['Не дубликаты', 'Дубликаты'],
            digits=4
        ))
        
        # Примеры ошибок
        print(f"\n{'='*80}")
        print("ПРИМЕРЫ ОШИБОК")
        print(f"{'='*80}")
        
        errors = []
        for i, (true_label, pred_label, prob) in enumerate(zip(
            best_metrics['y_true'], 
            best_metrics['y_pred'],
            best_metrics['y_pred_proba']
        )):
            if true_label != pred_label:
                errors.append({
                    'index': i,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'probability': prob,
                    'item': data[i]
                })
        
        if errors:
            print(f"\nВсего ошибок: {len(errors)}")
            print(f"\nПервые 5 ошибок:\n")
            
            for i, error in enumerate(errors[:5], 1):
                item = error['item']
                print(f"{i}. ID: {item['id']}")
                print(f"   Истинная метка: {'Дубликат' if error['true_label'] == 1 else 'Не дубликат'}")
                print(f"   Предсказание:   {'Дубликат' if error['pred_label'] == 1 else 'Не дубликат'}")
                print(f"   Вероятность дубликата: {error['probability']:.2%}")
                print(f"   Текст 1: {item['text1'][:100]}...")
                print(f"   Текст 2: {item['text2'][:100]}...")
                print()
        else:
            print("\n✅ Ошибок не обнаружено! Модель идеальна!")
    
    print(f"\n{'='*80}")
    print("✅ ВАЛИДАЦИЯ ЗАВЕРШЕНА")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

