# -*- coding: utf-8 -*-
"""关键结果回归测试：不重跑模型，只校验已生成产物的口径和数字。

运行：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
REPORT = ROOT / "outputs" / "报告"
DASH = ROOT / "outputs" / "看板"


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE / name, encoding="utf-8-sig")


class ProjectContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.transcript = pd.read_csv(
            ROOT / "data" / "clean" / "transcript.csv", encoding="utf-8-sig"
        )
        cls.user = pd.read_csv(
            ROOT / "data" / "clean" / "user_features.csv", encoding="utf-8-sig"
        )
        cls.offer_matrix = read_table("03_offer_matrix.csv")
        cls.offer_dataset = read_table("04_offer_dataset.csv")
        cls.segment_uplift = read_table("05_segment_uplift.csv")
        cls.deciles = read_table("05_uplift_deciles.csv")

    def test_event_counts(self) -> None:
        counts = self.transcript["event"].value_counts().to_dict()
        self.assertEqual(counts["transaction"], 138953)
        self.assertEqual(counts["offer received"], 76277)
        self.assertEqual(counts["offer viewed"], 57725)
        self.assertEqual(counts["offer completed"], 33579)
        self.assertEqual(len(self.transcript), 306534)

    def test_user_features(self) -> None:
        self.assertEqual(len(self.user), 17000)
        self.assertEqual(self.user.shape[1], 52)
        self.assertEqual(int(self.user["n_tx"].sum()), 138953)
        self.assertAlmostEqual(float(self.user["monetary"].sum()), 1775451.97, places=2)
        self.assertAlmostEqual(float(self.user["recency_h"].mean()), 91.43, places=2)
        self.assertEqual(int(self.user["n_tx"].eq(0).sum()), 422)

    def test_offer_matrix(self) -> None:
        self.assertEqual(len(self.offer_matrix), 54)
        self.assertEqual(int(self.offer_matrix["sends"].sum()), 76277)
        self.assertEqual(int(self.offer_matrix["comps"].sum()), 33579)

    def test_offer_dataset(self) -> None:
        self.assertEqual(len(self.offer_dataset), 76277)
        self.assertEqual(
            self.offer_dataset["split"].value_counts().to_dict(),
            {"train": 48850, "not_completable": 15235, "test": 12192},
        )
        self.assertEqual(int(self.offer_dataset["y_completed"].sum()), 33101)
        key = ["person", "offer_id", "send_time"]
        self.assertEqual(int(self.offer_dataset.duplicated(key).sum()), 0)

    def test_uplift_outputs(self) -> None:
        self.assertEqual(len(self.segment_uplift), 8)
        self.assertEqual(int(self.segment_uplift["n_sends"].sum()), 48877)
        self.assertEqual(int(self.segment_uplift["n_users"].sum()), 10469)
        self.assertEqual(len(self.deciles), 10)
        self.assertEqual(int(self.deciles["n"].sum()), 15215)

    def test_sql_validation_all_pass(self) -> None:
        validation = pd.read_csv(TABLE / "sql" / "_validation.csv", encoding="utf-8-sig")
        self.assertEqual(len(validation), 34)
        self.assertTrue((validation["pass"].astype(str).str.lower() == "true").all())
        policy = pd.read_csv(TABLE / "sql" / "09_policy_summary.csv", encoding="utf-8-sig").iloc[0]
        self.assertAlmostEqual(float(policy["inc_net"]), 119622, delta=2)
        self.assertAlmostEqual(float(policy["incremental_roi"]), 2.655, delta=0.01)

    def test_ab_power_pooled(self) -> None:
        ab = read_table("08_ab_power.csv")
        pooled = ab[ab["segment"].str.contains("pooled")].iloc[0]
        self.assertEqual(int(pooled["n_users"]), 5542)
        self.assertEqual(int(pooled["n_per_arm"]), 2771)
        self.assertAlmostEqual(float(pooled["delta_net_gmv"]), 5.16, delta=0.01)
        self.assertAlmostEqual(float(pooled["mde_net_gmv_80"]), 3.19, delta=0.01)
        self.assertAlmostEqual(float(pooled["required_n_per_arm_net_2_5"]), 4515, delta=1)

    def test_monitoring_contract(self) -> None:
        psi = read_table("09_psi.csv")
        score = psi[psi["category"] == "score"]
        self.assertLess(float(score["psi"].max()), 0.10)
        spend = psi[(psi["feature"] == "spend_prior") & (psi["category"] == "behavior_history")]
        self.assertGreater(float(spend["psi"].max()), 0.25)

        decay = read_table("09_model_decay.csv")
        ref = decay[decay["period"] == "D7-13"].iloc[0]
        later = decay[decay["period"].isin(["D14-20", "D21-29"])]
        self.assertTrue((later["auc"] >= float(ref["auc"])).all())
        self.assertTrue((later["brier"] <= float(ref["brier"])).all())

        coupon = read_table("09_coupon_monitor.csv")
        ref_coupon = coupon[coupon["period"] == "D7-13"].iloc[0]
        last = coupon[coupon["period"] == "D21-29"].iloc[0]
        self.assertGreater(float(last["net_gmv_per_send"]), float(ref_coupon["net_gmv_per_send"]))
        self.assertGreaterEqual(float(coupon["redemption_rate"].min()), 0.95)

        alerts = read_table("09_alerts.csv")
        self.assertEqual(len(alerts), 12)
        self.assertTrue((alerts["type"] == "PSI").all())
        self.assertTrue((alerts["category"] == "behavior_history").all())

    def test_reports_and_dashboard_exist(self) -> None:
        self.assertTrue((REPORT / "09_监控与稳定性.md").exists())
        self.assertTrue((DASH / "06_dashboard.html").exists())
        self.assertGreater((DASH / "06_dashboard.html").stat().st_size, 1_000_000)
        self.assertTrue((FIG / "09_psi_heatmap.png").exists())

        html = (DASH / "06_dashboard.html").read_text(encoding="utf-8")
        self.assertIn('data-view="validation"', html)
        self.assertIn('id="view-validation"', html)

        runner = (ROOT / "src" / "run_all.py").read_text(encoding="utf-8")
        self.assertLess(
            runner.index('"script": "09_monitoring.py"'),
            runner.index('"script": "06_build_dashboard.py"'),
            "06 看板必须在 07-09 验证产物之后构建",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
