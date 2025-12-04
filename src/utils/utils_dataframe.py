# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

import pandas as pd


# ============================ DATAFRAME UTILS ============================

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = (
            out[c].astype(str).str.strip()
            .replace({"nan": "", "NaN": "", "None": "", "none": "", "NULL": "", "null": ""})
        )
    return out


def select_latest_by_date(df: pd.DataFrame, side_value: str) -> pd.DataFrame:
    subset = df[df["Pre/Post"].str.lower() == side_value.lower()]
    if subset.empty:
        return subset
    if "Date" not in subset.columns or (subset["Date"].astype(str).str.len() == 0).all():
        return subset
    subset = subset.copy()
    subset["__Date_dt"] = pd.to_datetime(subset["Date"], format="%Y-%m-%d", errors="coerce")
    max_date = subset["__Date_dt"].max()
    return subset[subset["__Date_dt"] == max_date].drop(columns="__Date_dt")


def make_index_by_keys(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    dfx = df.copy()
    for c in keys:
        if c not in dfx.columns:
            dfx[c] = ""
    dfx["_join_key"] = dfx[keys].agg("||".join, axis=1)
    dfx = dfx.set_index("_join_key", drop=True)
    if dfx.index.has_duplicates:
        dfx = dfx[~dfx.index.duplicated(keep="last")]
    return dfx


def ensure_column_before(df: pd.DataFrame, col_to_move: str, before_col: str) -> pd.DataFrame:
    """
    Utility to keep a helper column immediately before another column in the Excel output.
    """
    if df is None or df.empty:
        return df
    if col_to_move in df.columns and before_col in df.columns:
        cols = list(df.columns)
        cols.remove(col_to_move)
        insert_pos = cols.index(before_col)
        cols.insert(insert_pos, col_to_move)
        df = df[cols]
    return df


def ensure_column_after(df: pd.DataFrame, col_to_move: str, after_col: str) -> pd.DataFrame:
    """
    Utility to keep a helper column immediately after another column in the Excel output.
    """
    if df is None or df.empty:
        return df
    if col_to_move in df.columns and after_col in df.columns:
        cols = list(df.columns)
        cols.remove(col_to_move)
        insert_pos = cols.index(after_col) + 1
        cols.insert(insert_pos, col_to_move)
        df = df[cols]
    return df


def drop_columns(df: pd.DataFrame, unwanted) -> pd.DataFrame:
    """
    Drop a list of unwanted columns if they exist; used to keep Excel output compact.
    """
    if df is None or df.empty:
        return df
    return df.drop(columns=[c for c in unwanted if c in df.columns], errors="ignore")
