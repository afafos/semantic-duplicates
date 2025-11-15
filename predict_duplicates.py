#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для использования обученных моделей определения дубликатов
Выводит предсказания от всех доступных моделей
"""

from train_duplicate_model import DuplicateNewsModel
from pathlib import Path

# Тексты для сравнения
text1 = "Китай увеличил импорт смартфонов в Россию"
text2 = "Китай увеличил импорт автомобилей в Россию"

# Вывод информации о текстах
print("=" * 80)
print("СРАВНЕНИЕ ТЕКСТОВ НА ДУБЛИКАТЫ")
print("=" * 80)
print(f"\n📝 Текст 1 ({len(text1)} символов):")
print(f"   {text1[:150]}..." if len(text1) > 150 else f"   {text1}")
print(f"\n📝 Текст 2 ({len(text2)} символов):")
print(f"   {text2[:150]}..." if len(text2) > 150 else f"   {text2}")

# Поиск всех доступных моделей
models_dir = Path(__file__).parent / "models"
if not models_dir.exists():
    print(f"\n❌ Ошибка: Папка с моделями не найдена: {models_dir}")
    print("   Сначала обучите модели: python train_duplicate_model.py")
    exit(1)

model_files = list(models_dir.glob("duplicate_model_*.joblib"))

if not model_files:
    print(f"\n❌ Ошибка: Не найдено обученных моделей в {models_dir}")
    print("   Сначала обучите модели: python train_duplicate_model.py")
    exit(1)

print(f"\n🔍 Найдено моделей: {len(model_files)}")
print("\n" + "=" * 80)
print("ПРЕДСКАЗАНИЯ ОТ ВСЕХ МОДЕЛЕЙ")
print("=" * 80)

# Словарь для хранения результатов
results = []

# Загружаем и тестируем каждую модель
for model_file in sorted(model_files):
    model_name = model_file.stem.replace("duplicate_model_", "").upper()
    
    try:
        # Загрузка модели
        model = DuplicateNewsModel()
        model.load(model_file)
        
        # Предсказание
        is_duplicate, probability = model.predict(text1, text2)
        
        results.append({
            'name': model_name,
            'is_duplicate': is_duplicate == 1,
            'prob_not_duplicate': probability[0],
            'prob_duplicate': probability[1]
        })
        
        # Вывод результата для этой модели
        print(f"\n{'='*80}")
        print(f"Модель: {model_name}")
        print(f"{'='*80}")
        
        if is_duplicate == 1:
            status = "🔄 ДУБЛИКАТ"
            emoji = "✅"
        else:
            status = "✅ НЕ ДУБЛИКАТ"
            emoji = "❌"
        
        print(f"Результат: {status}")
        print(f"\nВероятности:")
        
        # Визуализация вероятностей
        bar_length = 40
        not_dup_bar = "█" * int(probability[0] * bar_length)
        dup_bar = "█" * int(probability[1] * bar_length)
        
        print(f"  Не дубликат: {probability[0]:>6.2%} |{not_dup_bar:<{bar_length}}|")
        print(f"  Дубликат:    {probability[1]:>6.2%} |{dup_bar:<{bar_length}}|")
        
    except Exception as e:
        print(f"\n❌ Ошибка при загрузке модели {model_name}: {e}")

# Сводная таблица
print("\n" + "=" * 80)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 80)
print(f"\n{'Модель':<25} {'Результат':<20} {'Вер-ть не дубл.':<20} {'Вер-ть дубл.':<15}")
print("-" * 80)

for result in results:
    status = "🔄 ДУБЛИКАТ" if result['is_duplicate'] else "✅ НЕ ДУБЛИКАТ"
    print(f"{result['name']:<25} {status:<20} {result['prob_not_duplicate']:>6.2%}             {result['prob_duplicate']:>6.2%}")

# Консенсус моделей
if results:
    duplicates_count = sum(1 for r in results if r['is_duplicate'])
    not_duplicates_count = len(results) - duplicates_count
    
    print("\n" + "=" * 80)
    print("КОНСЕНСУС МОДЕЛЕЙ")
    print("=" * 80)
    print(f"Дубликат:     {duplicates_count} из {len(results)} моделей ({duplicates_count/len(results)*100:.1f}%)")
    print(f"Не дубликат:  {not_duplicates_count} из {len(results)} моделей ({not_duplicates_count/len(results)*100:.1f}%)")
    
    avg_prob_duplicate = sum(r['prob_duplicate'] for r in results) / len(results)
    print(f"\nСредняя вероятность дубликата: {avg_prob_duplicate:.2%}")
    
    if duplicates_count > len(results) / 2:
        print(f"\n🔄 ИТОГОВОЕ РЕШЕНИЕ: ДУБЛИКАТ")
    else:
        print(f"\n✅ ИТОГОВОЕ РЕШЕНИЕ: НЕ ДУБЛИКАТ")

print("\n" + "=" * 80)
