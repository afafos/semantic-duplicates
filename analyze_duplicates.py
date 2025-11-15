#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для анализа разметки семантических дубликатов новостей
Выполняет анализ с выводом в консоль
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def load_annotations(directory: Path) -> Dict[str, Dict]:
    """Загружает все файлы с разметкой"""
    annotations = {}
    
    for json_file in sorted(directory.glob("*.json")):
        annotator_name = json_file.stem
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            annotations[annotator_name] = {item['id']: item for item in data}
    
    return annotations


def get_label_stats(annotations_dict: Dict) -> Dict:
    """Подсчитывает статистику по разметкам"""
    stats = {
        'duplicates': 0,
        'not_duplicates': 0,
        'not_labeled': 0,
        'total': len(annotations_dict)
    }
    
    for item in annotations_dict.values():
        label = item.get('label', [])
        
        if not label:
            stats['not_labeled'] += 1
        elif 'are_duplicates' in label:
            stats['duplicates'] += 1
        elif 'not_duplicates' in label:
            stats['not_duplicates'] += 1
        else:
            stats['not_labeled'] += 1
    
    return stats


def calculate_agreement(annotations: Dict[str, Dict]) -> Dict:
    """Вычисляет согласованность между разметчиками"""
    annotators = list(annotations.keys())
    
    all_ids = set(annotations[annotators[0]].keys())
    for annotator in annotators[1:]:
        all_ids = all_ids.intersection(set(annotations[annotator].keys()))
    
    agreement_stats = {
        'common_ids_count': len(all_ids),
        'pairwise_agreement': {},
        'full_agreement': {'duplicates': 0, 'not_duplicates': 0, 'not_labeled': 0}
    }
    
    # Попарное сравнение
    for i, ann1 in enumerate(annotators):
        for j, ann2 in enumerate(annotators):
            if i >= j:
                continue
            
            pair_name = f"{ann1} vs {ann2}"
            common_ids = set(annotations[ann1].keys()).intersection(
                set(annotations[ann2].keys())
            )
            
            agree_count = 0
            disagree_count = 0
            both_labeled = 0
            agree_duplicates = 0
            agree_not_duplicates = 0
            
            for item_id in common_ids:
                label1 = annotations[ann1][item_id].get('label', [])
                label2 = annotations[ann2][item_id].get('label', [])
                
                l1 = label1[0] if label1 else None
                l2 = label2[0] if label2 else None
                
                if l1 and l2:
                    both_labeled += 1
                    if l1 == l2:
                        agree_count += 1
                        if l1 == 'are_duplicates':
                            agree_duplicates += 1
                        elif l1 == 'not_duplicates':
                            agree_not_duplicates += 1
                    else:
                        disagree_count += 1
            
            agreement_rate = (agree_count / both_labeled * 100) if both_labeled > 0 else 0
            
            agreement_stats['pairwise_agreement'][pair_name] = {
                'common_ids': len(common_ids),
                'both_labeled': both_labeled,
                'agree': agree_count,
                'disagree': disagree_count,
                'agreement_rate': agreement_rate,
                'agree_duplicates': agree_duplicates,
                'agree_not_duplicates': agree_not_duplicates
            }
    
    # Полное согласие всех
    for item_id in all_ids:
        labels = [annotations[ann][item_id].get('label', []) for ann in annotators]
        labels = [l[0] if l else None for l in labels]
        
        if all(l == labels[0] for l in labels) and labels[0]:
            if labels[0] == 'are_duplicates':
                agreement_stats['full_agreement']['duplicates'] += 1
            elif labels[0] == 'not_duplicates':
                agreement_stats['full_agreement']['not_duplicates'] += 1
        elif all(l is None for l in labels):
            agreement_stats['full_agreement']['not_labeled'] += 1
    
    return agreement_stats


def find_disagreements(annotations: Dict[str, Dict]) -> List[Dict]:
    """Находит случаи несогласия между разметчиками"""
    annotators = list(annotations.keys())
    disagreements = []
    
    all_ids = set(annotations[annotators[0]].keys())
    for annotator in annotators[1:]:
        all_ids = all_ids.union(set(annotations[annotator].keys()))
    
    for item_id in all_ids:
        labels_dict = {}
        
        for annotator in annotators:
            if item_id in annotations[annotator]:
                label = annotations[annotator][item_id].get('label', [])
                labels_dict[annotator] = label[0] if label else None
        
        labeled_values = [v for v in labels_dict.values() if v is not None]
        
        if len(labeled_values) >= 2 and len(set(labeled_values)) > 1:
            text = ""
            for annotator in annotators:
                if item_id in annotations[annotator]:
                    text = annotations[annotator][item_id].get('text', '')[:200]
                    break
            
            disagreements.append({
                'id': item_id,
                'labels': labels_dict,
                'text_preview': text
            })
    
    return disagreements


def print_statistics(annotations: Dict[str, Dict]):
    """Выводит подробную статистику"""
    print("=" * 80)
    print("СТАТИСТИКА ПО РАЗМЕТКАМ СЕМАНТИЧЕСКИХ ДУБЛИКАТОВ")
    print("=" * 80)
    print()
    
    # 1. Статистика по разметчикам
    print("1. СТАТИСТИКА ПО РАЗМЕТЧИКАМ")
    print("-" * 80)
    
    total_stats = defaultdict(int)
    
    for annotator, ann_data in sorted(annotations.items()):
        stats = get_label_stats(ann_data)
        
        for key in ['duplicates', 'not_duplicates', 'not_labeled', 'total']:
            total_stats[key] += stats[key]
        
        labeled = stats['duplicates'] + stats['not_duplicates']
        
        print(f"\n📝 {annotator}:")
        print(f"  Всего новостей: {stats['total']}")
        print(f"  ✅ Дубликаты: {stats['duplicates']} ({stats['duplicates']/stats['total']*100:.1f}%)")
        print(f"  ❌ НЕ дубликаты: {stats['not_duplicates']} ({stats['not_duplicates']/stats['total']*100:.1f}%)")
        print(f"  ⚪ Не размечено: {stats['not_labeled']} ({stats['not_labeled']/stats['total']*100:.1f}%)")
        print(f"  📊 Прогресс разметки: {labeled} / {stats['total']} ({labeled/stats['total']*100:.1f}%)")
    
    # Суммарная статистика
    print("\n" + "=" * 80)
    print("СУММАРНАЯ СТАТИСТИКА")
    print("-" * 80)
    print(f"Всего записей (суммарно по всем разметчикам): {total_stats['total']}")
    print(f"✅ Размечено как дубликаты: {total_stats['duplicates']}")
    print(f"❌ Размечено как НЕ дубликаты: {total_stats['not_duplicates']}")
    print(f"⚪ Не размечено: {total_stats['not_labeled']}")
    
    # 2. Согласованность
    print("\n" + "=" * 80)
    print("2. СОГЛАСОВАННОСТЬ МЕЖДУ РАЗМЕТЧИКАМИ")
    print("-" * 80)
    
    agreement = calculate_agreement(annotations)
    
    print(f"\nОбщих ID у всех разметчиков: {agreement['common_ids_count']}")
    
    print("\n📊 ПОПАРНОЕ СРАВНЕНИЕ:")
    print("-" * 80)
    
    for pair_name, pair_stats in sorted(agreement['pairwise_agreement'].items()):
        if pair_stats['both_labeled'] == 0:
            continue
        
        rate = pair_stats['agreement_rate']
        emoji = "🟢" if rate >= 80 else "🟡" if rate >= 70 else "🔴"
        
        print(f"\n{emoji} {pair_name}:")
        print(f"  Общих новостей: {pair_stats['common_ids']}")
        print(f"  Размечено обоими: {pair_stats['both_labeled']}")
        print(f"  Согласны: {pair_stats['agree']} ({rate:.1f}%)")
        print(f"    - оба: дубликаты = {pair_stats['agree_duplicates']}")
        print(f"    - оба: НЕ дубликаты = {pair_stats['agree_not_duplicates']}")
        print(f"  Не согласны: {pair_stats['disagree']}")
    
    print("\n" + "=" * 80)
    print("ПОЛНОЕ СОГЛАСИЕ ВСЕХ РАЗМЕТЧИКОВ")
    print("-" * 80)
    print(f"✅ Все согласны: дубликаты = {agreement['full_agreement']['duplicates']}")
    print(f"❌ Все согласны: НЕ дубликаты = {agreement['full_agreement']['not_duplicates']}")
    print(f"⚪ Все не разметили = {agreement['full_agreement']['not_labeled']}")
    
    total_agreement = (agreement['full_agreement']['duplicates'] + 
                      agreement['full_agreement']['not_duplicates'])
    if agreement['common_ids_count'] > 0:
        agreement_rate = total_agreement / agreement['common_ids_count'] * 100
        print(f"\n📈 Процент полного согласия: {agreement_rate:.1f}%")
    
    # 3. Конфликты
    print("\n" + "=" * 80)
    print("3. ПРИМЕРЫ КОНФЛИКТОВ В РАЗМЕТКАХ")
    print("-" * 80)
    
    disagreements = find_disagreements(annotations)
    print(f"\n🔍 Всего найдено конфликтов: {len(disagreements)}")
    
    if disagreements:
        print("\nПримеры (первые 10):\n")
        for i, conflict in enumerate(disagreements[:10], 1):
            print(f"{i}. ID: {conflict['id']}")
            print(f"   Разметки:")
            for annotator, label in sorted(conflict['labels'].items()):
                label_str = label if label else "-"
                emoji = "✅" if label == "are_duplicates" else "❌" if label == "not_duplicates" else "⚪"
                print(f"     {emoji} {annotator}: {label_str}")
            print(f"   Текст: {conflict['text_preview']}...")
            print()
    
    return disagreements


def find_agreed_annotations(annotations: Dict[str, Dict], 
                           annotator1: str, 
                           annotator2: str,
                           output_file: Path):
    """
    Находит новости, где два разметчика согласны, и сохраняет их в файл
    
    Args:
        annotations: словарь {имя_разметчика: {id: данные}}
        annotator1: имя первого разметчика
        annotator2: имя второго разметчика
        output_file: путь к выходному файлу
    """
    if annotator1 not in annotations or annotator2 not in annotations:
        print(f"❌ Ошибка: Один из разметчиков не найден!")
        return None, None
    
    agreed_items = []
    stats = {
        'total_compared': 0,
        'both_labeled': 0,
        'agreed': 0,
        'agreed_duplicates': 0,
        'agreed_not_duplicates': 0,
        'disagreed': 0
    }
    
    # Получаем общие ID
    common_ids = set(annotations[annotator1].keys()).intersection(
        set(annotations[annotator2].keys())
    )
    
    stats['total_compared'] = len(common_ids)
    
    for item_id in sorted(common_ids):
        label1 = annotations[annotator1][item_id].get('label', [])
        label2 = annotations[annotator2][item_id].get('label', [])
        
        l1 = label1[0] if label1 else None
        l2 = label2[0] if label2 else None
        
        if l1 and l2:
            stats['both_labeled'] += 1
            
            if l1 == l2:
                # Разметки совпадают
                stats['agreed'] += 1
                
                if l1 == 'are_duplicates':
                    stats['agreed_duplicates'] += 1
                elif l1 == 'not_duplicates':
                    stats['agreed_not_duplicates'] += 1
                
                # Добавляем в список согласованных
                item_data = annotations[annotator1][item_id].copy()
                item_data['agreed_label'] = l1
                item_data['annotators'] = [annotator1, annotator2]
                agreed_items.append(item_data)
            else:
                stats['disagreed'] += 1
    
    # Сохраняем в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(agreed_items, f, ensure_ascii=False, indent=2)
    
    # Выводим статистику
    print("\n" + "=" * 80)
    print(f"СОГЛАСОВАННЫЕ РАЗМЕТКИ: {annotator1} и {annotator2}")
    print("-" * 80)
    print(f"📊 Общих новостей: {stats['total_compared']}")
    print(f"📝 Размечено обоими: {stats['both_labeled']}")
    print(f"✅ Согласны: {stats['agreed']} ({stats['agreed']/stats['both_labeled']*100:.1f}%)")
    print(f"   - оба разметили как дубликаты: {stats['agreed_duplicates']}")
    print(f"   - оба разметили как НЕ дубликаты: {stats['agreed_not_duplicates']}")
    print(f"❌ Не согласны: {stats['disagreed']}")
    print(f"\n💾 Сохранено согласованных новостей: {len(agreed_items)}")
    print(f"📁 Файл: {output_file}")
    print("=" * 80)
    
    return agreed_items, stats


def main():
    """Основная функция"""
    script_dir = Path(__file__).parent
    annotations_dir = script_dir / "news_duplicates"
    
    if not annotations_dir.exists():
        print(f"❌ Ошибка: Директория {annotations_dir} не найдена!")
        return
    
    print("\n🔍 Загрузка данных...")
    annotations = load_annotations(annotations_dir)
    
    if not annotations:
        print("❌ Ошибка: Не найдено ни одного JSON файла!")
        return
    
    print(f"✅ Загружено файлов: {len(annotations)}")
    print(f"👥 Разметчики: {', '.join(sorted(annotations.keys()))}")
    print()
    
    # Анализ и вывод статистики
    print_statistics(annotations)
    
    # Сохранить согласованные разметки osokina-sa и popov-ny
    output_file = script_dir / "agreed_annotations_osokina_popov.json"
    find_agreed_annotations(annotations, 'osokina-sa', 'popov-ny', output_file)
    
    print("\n" + "=" * 80)
    print("✅ АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 80)


if __name__ == "__main__":
    main()

