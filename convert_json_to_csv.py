#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для конвертации agreed_annotations_osokina_popov.json в CSV
Парсит пары текстов и убирает колонку annotators
"""

import json
import csv
import pandas as pd
from pathlib import Path


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


def convert_json_to_csv(json_path, csv_path):
    """
    Конвертирует JSON файл в CSV
    
    Args:
        json_path: путь к входному JSON файлу
        csv_path: путь к выходному CSV файлу
    """
    print("=" * 80)
    print("КОНВЕРТАЦИЯ JSON В CSV")
    print("=" * 80)
    
    # 1. Загрузка JSON
    print(f"\n1. Загрузка данных из {json_path.name}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Загружено записей: {len(data)}")
    
    # 2. Парсинг данных
    print("\n2. Парсинг текстовых пар...")
    rows = []
    
    for idx, item in enumerate(data, start=1):
        text1, text2 = parse_text_pair(item['text'])
        
        if text1 is None or text2 is None:
            print(f"⚠️  Пропущена запись с ID {item['id']} (ошибка парсинга)")
            continue
        
        # Формируем строку: id (новая нумерация с 1), text1, text2, label
        row = {
            'id': idx,
            'text1': text1,
            'text2': text2,
            'label': item['agreed_label']
        }
        rows.append(row)
    
    print(f"✅ Успешно обработано: {len(rows)} записей")
    
    # 3. Создание DataFrame
    print("\n3. Создание DataFrame...")
    df = pd.DataFrame(rows)
    
    print(f"✅ DataFrame создан")
    print(f"   Строк: {len(df)}")
    print(f"   Колонок: {len(df.columns)}")
    print(f"   Колонки: {list(df.columns)}")
    
    # Статистика по меткам
    print(f"\n📊 Распределение по label:")
    for label, count in df['label'].value_counts().items():
        print(f"   {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # 4. Сохранение в CSV
    print(f"\n4. Сохранение в CSV: {csv_path.name}...")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig', sep=';', 
              quoting=csv.QUOTE_ALL, escapechar='\\')
    
    print(f"✅ Файл сохранен!")
    print(f"   Размер файла: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 5. Пример данных
    print("\n" + "=" * 80)
    print("ПРИМЕР ПЕРВЫХ 3 ЗАПИСЕЙ")
    print("=" * 80)
    print(df.head(3).to_string())
    
    print("\n" + "=" * 80)
    print("✅ КОНВЕРТАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 80)
    print(f"\n📁 Входной файл: {json_path}")
    print(f"📁 Выходной файл: {csv_path}")


def main():
    """Основная функция"""
    script_dir = Path(__file__).parent
    
    # Пути к файлам
    json_path = script_dir / "agreed_annotations_osokina_popov.json"
    csv_path = script_dir / "agreed_annotations_osokina_popov.csv"
    
    # Проверка наличия входного файла
    if not json_path.exists():
        print(f"❌ Ошибка: Файл не найден: {json_path}")
        return
    
    # Конвертация
    convert_json_to_csv(json_path, csv_path)


if __name__ == "__main__":
    main()

