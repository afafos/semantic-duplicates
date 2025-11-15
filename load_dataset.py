# load_dataset.py
# -*- coding: utf-8 -*-
"""
Собирает единый размеченный датасет пар текстов для задачи дубликатов новостей.
Источники: merionum/ru_paraphraser, GEM/opusparcus (RU), cointegrated/ru-paraphrase-NMT-Leipzig.
Выход: Parquet (+CSV опционально).

Зависимости:
  pip install -U datasets pandas pyarrow huggingface_hub python-dotenv

python load_dataset.py --save ./unified_news_pairs.parquet --csv

"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# ──────────────────────────────────────────────────────────────────────────────
# .env / токены (только HF)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def setup_auth():
    """Загружает .env (HF_TOKEN) и логинит Hugging Face при наличии токена."""
    if load_dotenv is not None:
        env_path = Path(__file__).with_name(".env")
        if env_path.exists():
            load_dotenv(env_path)
            print(f">> Loaded .env from {env_path}")
        else:
            print(">> .env не найден — продолжу с переменными окружения")
    else:
        print(">> python-dotenv не установлен; пропускаю загрузку .env")

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print(">> Hugging Face: authenticated via HF_TOKEN")
        except Exception as e:
            print(f">> Hugging Face login failed: {e}. Продолжу анонимно там, где можно.")
    else:
        print(">> HF_TOKEN не задан — датасеты с HF буду тянуть анонимно (если публичные)")

# ──────────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────────
def _binary_label_series(
    vals, positive_values: Iterable, negative_values: Iterable, neutral_values: Iterable = ()
):
    """Маппит произвольные метки в {0,1}; нейтрали -> None (потом выкинем)."""
    pos = set(positive_values); neg = set(negative_values); neu = set(neutral_values)
    def _map(v):
        if v in pos: return 1
        if v in neg: return 0
        if v in neu: return None
        return None
    return pd.Series([_map(v) for v in vals])


def _normalize_columns(
    df: pd.DataFrame, col1_candidates, col2_candidates, label_candidates=None
) -> Tuple[pd.DataFrame, str, str, Optional[str]]:
    """Пытается найти колонки text1/text2/label по наборам кандидатов."""
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(cands):
        for c in cands or []:
            if isinstance(c, str) and c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None

    c1 = pick(col1_candidates)
    c2 = pick(col2_candidates)
    cl = pick(label_candidates)

    # Фолбэки
    if c1 is None or c2 is None:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if len(text_cols) >= 2: c1, c2 = text_cols[:2]
        else: raise ValueError("Не удалось определить колонки с текстом")

    if cl is None:
        for c in df.columns:
            if c.lower() in {"label","class","target","gold_label","is_duplicate","is_paraphrase"}:
                cl = c; break
        if cl is None:
            small_card = [c for c in df.columns if df[c].nunique(dropna=False) <= 5]
            cl = small_card[0] if small_card else None

    return df, c1, c2, cl


def _finalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.dropna(subset=["text1", "text2", "label"])
    df = df.astype({"label": "int8"})
    df["source"] = source
    # Удалим дубликаты без учёта порядка (A,B) ~ (B,A)
    key = df.apply(lambda r: tuple(sorted((r["text1"], r["text2"]))), axis=1)
    df = df.loc[~key.duplicated()].reset_index(drop=True)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Лоадеры HF
# ──────────────────────────────────────────────────────────────────────────────
def load_paraphraser(include_near_as_positive: bool = False) -> pd.DataFrame:
    """
    merionum/ru_paraphraser
      метки бывают числовые (1/0/-1) ИЛИ строковые ('1','0','-1' и пр.) — обрабатываем оба случая.
      1=точный парафраз, 0=near (по флагу либо позитив, либо выбрасываем), -1=не парафраз.
    """
    print(">> Загрузка ru_paraphraser ...")
    ds = load_dataset("merionum/ru_paraphraser")  # train/test
    frames = []
    for split in ("train", "test"):
        d = ds[split].to_pandas()
        d, c1, c2, cl = _normalize_columns(
            d,
            ["text1", "text_1", "sentence1", "sent1"],
            ["text2", "text_2", "sentence2", "sent2"],
            ["label", "class"],
        )

        lbl = d[cl]

        # 1) пробуем как числа
        lbl_num = pd.to_numeric(lbl, errors="coerce")
        if lbl_num.notna().mean() > 0.7:
            pos_vals = {1}
            neg_vals = {-1}
            near_vals = {0}
            if include_near_as_positive:
                pos_vals |= near_vals
                near_vals = set()
            y = _binary_label_series(lbl_num.astype(int), pos_vals, neg_vals, neutral_values=near_vals)
        else:
            # 2) строковый вариант
            s = lbl.astype(str).str.lower().str.strip()
            pos_vals = {"1", "paraphrase", "yes", "true", "точный", "парафраз"}
            neg_vals = {"-1", "no", "false", "не парафраз", "not paraphrase", "non-paraphrase"}
            near_vals = {"0", "near", "близкий"}
            if include_near_as_positive:
                pos_vals |= near_vals
                near_vals = set()

            def map_str(v):
                if v in pos_vals: return 1
                if v in neg_vals: return 0
                if v in near_vals: return None
                try:
                    f = float(v)
                    if f == 1: return 1
                    if f == 0: return None if not include_near_as_positive else 1
                    if f == -1: return 0
                except Exception:
                    pass
                return None

            y = s.map(map_str)

        out = pd.DataFrame(
            {"text1": d[c1].astype(str), "text2": d[c2].astype(str), "label": y}
        ).dropna(subset=["label"])

        if out.empty:
            uniq_preview = pd.Series(lbl).astype(str).str.slice(0, 20).value_counts().head(5).to_dict()
            print(f">> ВНИМАНИЕ: ru_paraphraser[{split}] дал 0 строк после маппинга. "
                  f"Примеры меток: {uniq_preview}")

        frames.append(_finalize(out, source="ru_paraphraser"))

    res = pd.concat(frames, ignore_index=True)
    if res.empty:
        raise RuntimeError("ru_paraphraser: не удалось распарсить метки — проверь схему датасета/версии пакетов.")
    return res


def load_opusparcus_ru(pos_threshold: float = 3.0, neg_threshold: float = 2.0) -> pd.DataFrame:
    """
    GEM/opusparcus (ru): грузим готовые JSONL без dataset script.
    Берём validation/test, фильтруем RU и маппим annot_score -> {0,1}:
      <=2 -> 0; >=3 -> 1 (границы настраиваемы).
    """
    print(">> Загрузка opusparcus_ru ...")
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
    frames = []
    for split in ("validation", "test"):
        d = ds[split].to_pandas()
        if "lang" in d.columns:
            d = d[d["lang"] == "ru"].copy()
        d["annot_score"] = pd.to_numeric(d["annot_score"], errors="coerce")
        keep = (d["annot_score"] >= pos_threshold) | (d["annot_score"] <= neg_threshold)
        pos  = (d["annot_score"] >= pos_threshold)
        tmp = pd.DataFrame({
            "text1": d.loc[keep, "sent1"].astype(str),
            "text2": d.loc[keep, "sent2"].astype(str),
            "label": pos.loc[keep].astype("int8"),
        })
        frames.append(_finalize(tmp, source="opusparcus_ru"))
    return pd.concat(frames, ignore_index=True)


def load_ru_paraphrase_nmt_leipzig(
    min_p_good: Optional[float] = 0.85, min_labse: float = 0.88
) -> pd.DataFrame:
    """
    cointegrated/ru-paraphrase-NMT-Leipzig: синтетические позитивы.
    Фильтруем либо по p_good (если есть), либо по labse_sim.
    """
    print(">> Загрузка ru_paraphrase_nmt_leipzig ...")
    ds = load_dataset(
        "cointegrated/ru-paraphrase-NMT-Leipzig",
        data_files={"train":"train.csv","val":"val.csv","test":"test.csv"},
    )
    frames = []
    for split in ("train","val","test"):
        d = ds[split].to_pandas()
        if "p_good" in d.columns and min_p_good is not None:
            d = d[pd.to_numeric(d["p_good"], errors="coerce") >= min_p_good]
        else:
            d = d[pd.to_numeric(d["labse_sim"], errors="coerce") >= min_labse]
        out = pd.DataFrame({
            "text1": d["original"].astype(str),
            "text2": d["ru"].astype(str),
            "label": 1,
        })
        frames.append(_finalize(out, source="ru_paraphrase_nmt_leipzig"))
    return pd.concat(frames, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# Оркестратор
# ──────────────────────────────────────────────────────────────────────────────
def build_unified_dataset(
    include_paraphraser: bool = True,
    include_opusparcus: bool = True,
    include_leipzig: bool = True,
    include_near_as_positive: bool = False,  # по умолчанию near НЕ считаем позитивом
    save_path: str = "./unified_news_pairs.parquet",
    also_save_csv: bool = False,
) -> pd.DataFrame:
    parts = []
    if include_paraphraser:
        parts.append(load_paraphraser(include_near_as_positive))
    if include_opusparcus:
        parts.append(load_opusparcus_ru())
    if include_leipzig:
        parts.append(load_ru_paraphrase_nmt_leipzig())

    if not parts:
        raise RuntimeError("Нечего объединять: все источники отключены.")

    data = pd.concat(parts, ignore_index=True)
    data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Короткий отчёт
    print("\nUnified dataset summary:")
    print(
        data.groupby(["source"])["label"]
        .agg(["count","mean"])
        .rename(columns={"mean":"positives_rate"})
        .sort_values(by="count", ascending=False)
    )

    # Сохранение
    save_path = Path(save_path)
    data.to_parquet(save_path, index=False)
    if also_save_csv:
        data.to_csv(save_path.with_suffix(".csv"), index=False)
    print(f"\nSaved to {save_path.resolve()}")
    if also_save_csv:
        print(f"Also saved CSV to {save_path.with_suffix('.csv').resolve()}")

    return data


def parse_args():
    p = argparse.ArgumentParser(description="Собрать единый датасет пар текстов для дедупликации новостей (без Kaggle)")
    p.add_argument("--no-paraphraser", action="store_true", help="не включать merionum/ru_paraphraser")
    p.add_argument("--no-opusparcus", action="store_true", help="не включать GEM/opusparcus (RU)")
    p.add_argument("--no-leipzig", action="store_true", help="не включать cointegrated/ru-paraphrase-NMT-Leipzig")
    p.add_argument(
        "--near-as-positive",
        action="store_true",
        help="считать label=0 (near) в ParaPhraser как позитив (по умолчанию ВЫКЛ)",
    )
    p.add_argument("--save", type=str, default="./unified_news_pairs.parquet", help="путь к выходному Parquet")
    p.add_argument("--csv", action="store_true", help="дополнительно сохранить CSV")
    return p.parse_args()


if __name__ == "__main__":
    setup_auth()
    args = parse_args()
    build_unified_dataset(
        include_paraphraser=not args.no_paraphraser,
        include_opusparcus=not args.no_opusparcus,
        include_leipzig=not args.no_leipzig,
        include_near_as_positive=args.near_as_positive,
        save_path=args.save,
        also_save_csv=args.csv,
    )
