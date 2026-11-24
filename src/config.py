# -*- coding: utf-8 -*-
"""项目统一路径配置：所有脚本都从这里取目录，避免各自拼路径。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"

OUT = ROOT / "outputs"
FIG = OUT / "figures"
TABLE = OUT / "tables"
REPORT = OUT / "报告"
DASH = OUT / "看板"
LOG = OUT / "logs"

ASSETS = ROOT / "src" / "assets"

for _d in (RAW, CLEAN, FIG, TABLE, REPORT, DASH, LOG, ASSETS):
    _d.mkdir(parents=True, exist_ok=True)
