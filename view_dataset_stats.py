#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для просмотра статистики по каждому исходному датасету.
Показывает количество записей, размеченных данных и распределение меток.

Датасеты:
  - merionum/ru_paraphraser (HuggingFace)
  - GEM/opusparcus (RU) (HuggingFace)
  - cointegrated/ru-paraphrase-NMT-Leipzig (HuggingFace)
  - viacheslavshalamov/russian-news-paraphrases-2020 (Kaggle)

Использование:
    python view_dataset_stats.py
"""

from __future__ import annotations
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from kaggle.api.kaggle_api_extended import KaggleApi


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты для аутентификации
# ──────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None


def setup_auth():
    """Настройка аутентификации для HuggingFace и Kaggle."""
    if _load_dotenv is not None:
        env_path = Path(__file__).with_name(".env")
        if env_path.exists():
            _load_dotenv(env_path)
    
    # HF authentication
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass
    
    # Kaggle authentication
    kaggle_json_path = Path(__file__).with_name("kaggle.json")
    if kaggle_json_path.exists():
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        target = kaggle_dir / "kaggle.json"
        if not target.exists():
            shutil.copy(kaggle_json_path, target)
            target.chmod(0o600)


# ──────────────────────────────────────────────────────────────────────────────
# Функции для загрузки и анализа датасетов
# ──────────────────────────────────────────────────────────────────────────────

def print_separator(char="=", length=80):
    """Печатает разделитель."""
    print(char * length)


def print_dataset_stats(name: str, df: pd.DataFrame, label_info: Optional[dict] = None):
    """
    Печатает статистику по датасету.
    
    Args:
        name: Название датасета
        df: DataFrame с исходными данными (до обработки)
        label_info: Дополнительная информация о метках
    """
    print_separator()
    print(f"📊 {name}")
    print_separator()
    
    total_records = len(df)
    print(f"Всего записей: {total_records:,}")
    
    # Информация о разметке меток
    if label_info:
        print(f"\nИнформация о метках:")
        for key, value in label_info.items():
            print(f"  {key}: {value}")
    
    print()


def analyze_ru_paraphraser():
    """Анализ датасета merionum/ru_paraphraser."""
    try:
        print("\n🔍 Загрузка merionum/ru_paraphraser...")
        ds = load_dataset("merionum/ru_paraphraser")
        
        all_data = []
        for split in ("train", "test"):
            d = ds[split].to_pandas()
            d['split'] = split
            all_data.append(d)
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Анализ меток
        label_col = None
        for col in ['label', 'class', 'gold_label']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            label_col = df.columns[2] if len(df.columns) > 2 else df.columns[-1]
        
        # Подсчёт меток
        labels = df[label_col]
        label_counts = labels.value_counts().sort_index()
        
        label_info = {
            "Колонка с метками": label_col,
            "Разметка": "1 = точный парафраз (дубликат), 0 = near (похожий), -1 = не парафраз",
            "Всего размечено": f"{len(labels):,} (100%)",
            ""  : "",
            "Распределение меток": ""
        }
        
        for label_val, count in label_counts.items():
            percentage = (count / len(labels)) * 100
            label_name = {
                1: "точный парафраз (дубликат)",
                0: "near (похожий)",
                -1: "не парафраз"
            }.get(label_val, f"метка {label_val}")
            label_info[f"  {label_val} ({label_name})"] = f"{count:,} ({percentage:.2f}%)"
        
        # Статистика по сплитам
        split_stats = df['split'].value_counts()
        label_info["  "] = ""
        label_info["По сплитам"] = ""
        for split, count in split_stats.items():
            percentage = (count / len(df)) * 100
            label_info[f"  {split}"] = f"{count:,} ({percentage:.2f}%)"
        
        print_dataset_stats("merionum/ru_paraphraser", df, label_info)
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке ru_paraphraser: {e}")
        print()


def analyze_opusparcus():
    """Анализ датасета GEM/opusparcus (RU)."""
    try:
        print("🔍 Загрузка GEM/opusparcus (RU)...")
        val_path = hf_hub_download(
            repo_id="GEM/opusparcus",
            filename="validation.jsonl",
            repo_type="dataset",
            token=False,
        )
        test_path = hf_hub_download(
            repo_id="GEM/opusparcus",
            filename="test.jsonl",
            repo_type="dataset",
            token=False,
        )
        
        ds = load_dataset("json", data_files={"validation": val_path, "test": test_path})
        
        all_data = []
        for split in ("validation", "test"):
            d = ds[split].to_pandas()
            d['split'] = split
            all_data.append(d)
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Фильтруем только русский язык
        df_all = df.copy()
        if "lang" in df.columns:
            df = df[df["lang"] == "ru"].copy()
        
        # Анализ меток (annot_score)
        df["annot_score"] = pd.to_numeric(df["annot_score"], errors="coerce")
        score_counts = df["annot_score"].value_counts().sort_index()
        
        label_info = {
            "Всего записей (все языки)": f"{len(df_all):,}",
            "Записей на русском": f"{len(df):,} (100% для анализа)",
            "Колонка с метками": "annot_score",
            "Разметка": "1.0-2.0 = не дубликат, 3.0-4.0 = дубликат, 2.5 исключается",
            "": "",
            "Распределение annot_score": ""
        }
        
        for score, count in score_counts.items():
            percentage = (count / len(df)) * 100
            if score <= 2.0:
                category = "(не дубликат)"
            elif score >= 3.0:
                category = "(дубликат)"
            else:
                category = "(исключается)"
            label_info[f"  {score:.1f} {category}"] = f"{count:,} ({percentage:.2f}%)"
        
        # После применения фильтров
        keep = (df["annot_score"] >= 3.0) | (df["annot_score"] <= 2.0)
        df_filtered = df[keep]
        pos = (df["annot_score"] >= 3.0).sum()
        neg = (df["annot_score"] <= 2.0).sum()
        excluded = len(df) - len(df_filtered)
        
        label_info["  "] = ""
        label_info["После применения порогов"] = ""
        label_info["  Дубликаты (score >= 3.0)"] = f"{pos:,} ({pos/len(df)*100:.2f}%)"
        label_info["  Не дубликаты (score <= 2.0)"] = f"{neg:,} ({neg/len(df)*100:.2f}%)"
        label_info["  Исключено (2.0 < score < 3.0)"] = f"{excluded:,} ({excluded/len(df)*100:.2f}%)"
        
        # Статистика по сплитам
        split_stats = df['split'].value_counts()
        label_info["   "] = ""
        label_info["По сплитам (RU)"] = ""
        for split, count in split_stats.items():
            percentage = (count / len(df)) * 100
            label_info[f"  {split}"] = f"{count:,} ({percentage:.2f}%)"
        
        print_dataset_stats("GEM/opusparcus (RU)", df, label_info)
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке opusparcus: {e}")
        print()


def analyze_leipzig():
    """Анализ датасета cointegrated/ru-paraphrase-NMT-Leipzig."""
    try:
        print("🔍 Загрузка cointegrated/ru-paraphrase-NMT-Leipzig...")
        ds = load_dataset(
            "cointegrated/ru-paraphrase-NMT-Leipzig",
            data_files={"train": "train.csv", "val": "val.csv", "test": "test.csv"},
        )
        
        all_data = []
        for split in ("train", "val", "test"):
            d = ds[split].to_pandas()
            d['split'] = split
            all_data.append(d)
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Анализ метрик качества
        has_p_good = "p_good" in df.columns
        has_labse = "labse_sim" in df.columns
        
        label_info = {
            "Тип датасета": "Синтетический (только дубликаты)",
            "Метод": "Перевод RU -> EN -> RU через NMT",
            "Всего размечено": f"{len(df):,} (100%, все дубликаты)",
            "": "",
            "Метрики качества": ""
        }
        
        if has_p_good:
            p_good = pd.to_numeric(df["p_good"], errors="coerce")
            label_info["  p_good (вероятность качества)"] = f"среднее: {p_good.mean():.3f}, мин: {p_good.min():.3f}, макс: {p_good.max():.3f}"
            
            # Распределение по порогам
            thresholds = [0.7, 0.8, 0.85, 0.9, 0.95]
            for th in thresholds:
                count = (p_good >= th).sum()
                pct = count / len(df) * 100
                label_info[f"    >= {th}"] = f"{count:,} ({pct:.2f}%)"
        
        if has_labse:
            labse_sim = pd.to_numeric(df["labse_sim"], errors="coerce")
            label_info["  labse_sim (LaBSE сходство)"] = f"среднее: {labse_sim.mean():.3f}, мин: {labse_sim.min():.3f}, макс: {labse_sim.max():.3f}"
            
            # Распределение по порогам
            thresholds = [0.7, 0.8, 0.85, 0.88, 0.9, 0.95]
            for th in thresholds:
                count = (labse_sim >= th).sum()
                pct = count / len(df) * 100
                label_info[f"    >= {th}"] = f"{count:,} ({pct:.2f}%)"
        
        # Статистика по сплитам
        split_stats = df['split'].value_counts()
        label_info["  "] = ""
        label_info["По сплитам"] = ""
        for split, count in split_stats.items():
            percentage = (count / len(df)) * 100
            label_info[f"  {split}"] = f"{count:,} ({percentage:.2f}%)"
        
        print_dataset_stats("cointegrated/ru-paraphrase-NMT-Leipzig", df, label_info)
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке ru-paraphrase-NMT-Leipzig: {e}")
        print()


def analyze_kaggle_news():
    """Анализ датасета viacheslavshalamov/russian-news-paraphrases-2020."""
    try:
        print("🔍 Загрузка viacheslavshalamov/russian-news-paraphrases-2020 (Kaggle)...")
        
        # Инициализируем Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Создаём временную директорию для загрузки
        download_dir = Path(__file__).parent / ".kaggle_cache"
        download_dir.mkdir(exist_ok=True)
        
        dataset_path = download_dir / "russian-news-paraphrases-2020"
        
        # Скачиваем датасет, если его ещё нет
        if not dataset_path.exists():
            print(f"  Скачивание в {dataset_path}...")
            api.dataset_download_files(
                "viacheslavshalamov/russian-news-paraphrases-2020",
                path=str(dataset_path),
                unzip=True
            )
        
        # Ищем XML файл с парафразами
        xml_file = dataset_path / "Russian-news-paraphrases-2020.xml"
        
        if not xml_file.exists():
            raise FileNotFoundError(f"Не найден файл {xml_file}")
        
        # Парсим XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Собираем статистику
        total_pairs = 0
        label_counts = {0: 0, 1: 0}
        has_title = 0
        has_text = 0
        
        for paraphrase in root.findall(".//paraphrase"):
            entry = {}
            for value in paraphrase.findall("value"):
                name = value.get("name")
                text = value.text or ""
                entry[name] = text
            
            if "class" in entry:
                try:
                    label = int(entry["class"])
                    if label in {0, 1}:
                        total_pairs += 1
                        label_counts[label] += 1
                        
                        if "title_1" in entry and "title_2" in entry:
                            if entry["title_1"] and entry["title_2"]:
                                has_title += 1
                        
                        if "text_1" in entry and "text_2" in entry:
                            if entry["text_1"] and entry["text_2"]:
                                has_text += 1
                except (ValueError, KeyError):
                    continue
        
        # Формируем информацию
        label_info = {
            "Формат": "XML файл с новостными заголовками 2020 года",
            "Всего размечено": f"{total_pairs:,} (100%)",
            "": "",
            "Доступные поля": "",
            "  Заголовки (title_1, title_2)": f"{has_title:,} пар ({has_title/total_pairs*100:.2f}%)",
            "  Полные тексты (text_1, text_2)": f"{has_text:,} пар ({has_text/total_pairs*100:.2f}%)",
            "  ": "",
            "Распределение меток": "",
        }
        
        for label, count in sorted(label_counts.items()):
            percentage = (count / total_pairs) * 100
            label_name = "парафраз (дубликат)" if label == 1 else "не парафраз"
            label_info[f"  {label} ({label_name})"] = f"{count:,} ({percentage:.2f}%)"
        
        # Создаём фиктивный DataFrame для вызова print_dataset_stats
        df = pd.DataFrame({"dummy": range(total_pairs)})
        
        print_dataset_stats("viacheslavshalamov/russian-news-paraphrases-2020 (Kaggle)", df, label_info)
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке Kaggle датасета: {e}")
        print()


def main():
    """Главная функция."""
    print("\n" + "=" * 80)
    print("СТАТИСТИКА ПО ИСХОДНЫМ ДАТАСЕТАМ")
    print("=" * 80)
    
    # Настраиваем аутентификацию
    setup_auth()
    
    # Анализируем каждый датасет
    analyze_ru_paraphraser()
    analyze_opusparcus()
    analyze_leipzig()
    analyze_kaggle_news()
    
    print("=" * 80)
    print("✅ Анализ завершён")
    print("=" * 80)


if __name__ == "__main__":
    main()

