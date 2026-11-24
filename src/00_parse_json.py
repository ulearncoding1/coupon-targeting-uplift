# -*- coding: utf-8 -*-
"""
00_parse_json.py —— 把 Starbucks Promotions 三个原始 JSON 转成干净的 CSV

原始数据特点（踩坑点）：
1. 三个 json 都是 JSONL 格式（每行一个对象），不是一个大数组，
   直接 pd.read_json(path) 会报 "Extra data" 错误，必须 lines=True。
2. transcript 里的券 id 键名带空格："offer id"，不是 offer_id。
3. profile 里 age=118 是缺失值标记（约 2175 个用户），需要转成 NaN。
4. profile 的 became_member_on 是 20180726 这种整数，要转成日期。

输出：
  data/clean/portfolio.csv   10 行，券的配置
  data/clean/profile.csv     17000 行，用户画像
  data/clean/transcript.csv  306534 行，事件流水（已拆出 offer_id / reward / amount）
"""

import os
import pandas as pd

from config import RAW, CLEAN

OUT = CLEAN


def read_jsonl(name):
    """读取 JSONL 文件（每行一个 JSON 对象）"""
    path = os.path.join(RAW, name)
    return pd.read_json(path, lines=True, encoding="utf-8")


# ---------- 1. portfolio：券配置 ----------
portfolio = read_jsonl("portfolio.json")
# channels 是列表，拆成三个 0/1 列，后面建模好直接用
for ch in ["email", "mobile", "social"]:
    portfolio[f"ch_{ch}"] = portfolio["channels"].apply(lambda x: int(ch in x))
portfolio = portfolio.drop(columns=["channels"])
portfolio.to_csv(os.path.join(OUT, "portfolio.csv"), index=False, encoding="utf-8-sig")
print(f"[OK] portfolio.csv   {portfolio.shape[0]} 行 × {portfolio.shape[1]} 列")


# ---------- 2. profile：用户画像 ----------
profile = read_jsonl("profile.json")
profile["age"] = profile["age"].replace(118, pd.NA)          # 118 = 缺失标记
profile["gender"] = profile["gender"].fillna("Unknown")      # None -> Unknown
profile["became_member_on"] = pd.to_datetime(
    profile["became_member_on"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
)
profile.to_csv(os.path.join(OUT, "profile.csv"), index=False, encoding="utf-8-sig")
print(f"[OK] profile.csv     {profile.shape[0]} 行 × {profile.shape[1]} 列")


# ---------- 3. transcript：事件流水 ----------
transcript = read_jsonl("transcript.json")
value = transcript["value"].apply(pd.Series)                 # 把 dict 摊平成列
transcript = pd.concat([transcript.drop(columns=["value"]), value], axis=1)

# 坑：原始数据里同一个字段有两种写法 —— "offer id"（带空格）和 "offer_id"，
# 摊平后会变成两列，必须合并成一列，否则后面 transcript["offer_id"] 会拿到 DataFrame。
if "offer id" in transcript.columns:
    if "offer_id" in transcript.columns:
        transcript["offer_id"] = transcript["offer_id"].fillna(transcript["offer id"])
        transcript = transcript.drop(columns=["offer id"])
    else:
        transcript = transcript.rename(columns={"offer id": "offer_id"})

# time 是"距实验开始的第几小时"，补一个 day 列方便做时机分析
transcript["day"] = (transcript["time"] // 24).astype("int64")
transcript = transcript[["person", "event", "time", "day", "offer_id", "reward", "amount"]]
assert transcript.columns.duplicated().sum() == 0, "存在重复列名"
transcript.to_csv(os.path.join(OUT, "transcript.csv"), index=False, encoding="utf-8-sig")
print(f"[OK] transcript.csv  {transcript.shape[0]} 行 × {transcript.shape[1]} 列")

# ---------- 4. 体检报告 ----------
print("\n事件类型分布：")
print(transcript["event"].value_counts().to_string())
print("\n券类型分布：")
print(portfolio["offer_type"].value_counts().to_string())
print("\n基础校验：")
offer_events = transcript["offer_id"].notna()
unknown = (~transcript.loc[offer_events, "offer_id"].isin(portfolio["id"])).sum()
print(f"  带券 id 的事件数：{offer_events.sum()}，其中未知券 id：{unknown}")
print(f"  transcript 覆盖用户数：{transcript['person'].nunique()} / profile {profile['id'].nunique()}")
print(f"  缺失人口属性的用户数：{profile['age'].isna().sum()}")
