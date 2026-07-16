import numpy as np
import polars as pl
from conftest import synthetic_hourly_matrix, utc

from grounded_weather_forecast.dataset.alignment import (
    alignment_study,
    load_recommended,
    recommended_semantics,
    write_alignment,
)


def semantics_matrix(n=200, seed=0):
    """A matrix where 'instprov' tracks inst truth and 'meanprov' tracks mean."""
    rng = np.random.default_rng(seed)
    inst_truth = rng.normal(15.0, 6.0, n)
    mean_truth = inst_truth + rng.normal(0.0, 3.0, n)  # decorrelated component
    base = synthetic_hourly_matrix(days=1, max_lead=1).head(0)  # schema only
    frame = pl.DataFrame(
        {
            "issue_time": [utc(2026, 1, 1)] * n,
            "valid_time": [utc(2026, 1, 1, 1)] * n,
            "lead_hours": [1.0] * n,
            "lead_bucket": ["1-3h"] * n,
            "source_kind": ["live"] * n,
            "fx__instprov__temp_c": inst_truth + rng.normal(0, 0.2, n),
            "fx__meanprov__temp_c": mean_truth + rng.normal(0, 0.2, n),
            "t__temp_c__inst": inst_truth,
            "t__temp_c__mean": mean_truth,
        }
    )
    assert base.width > 0  # keep the helper honest about schema reuse
    return frame


class TestAlignmentStudy:
    def test_detects_native_semantics(self):
        study = alignment_study(semantics_matrix())
        by_source = {row["source"]: row for row in study.to_dicts()}
        assert by_source["instprov"]["preferred"] == "inst"
        assert by_source["meanprov"]["preferred"] == "mean"
        assert by_source["instprov"]["r_inst"] > by_source["instprov"]["r_mean"]

    def test_thin_data_undetermined(self):
        study = alignment_study(semantics_matrix(n=20))
        assert study["preferred"].null_count() == study.height

    def test_recommendation_majority(self):
        study = pl.DataFrame(
            {
                "variable": ["temp_c"] * 3,
                "source": ["a", "b", "c"],
                "r_inst": [0.9, 0.8, 0.7],
                "r_mean": [0.8, 0.9, 0.9],
                "n": [100, 400, 400],
                "preferred": ["inst", "mean", "mean"],
            }
        )
        assert recommended_semantics(study) == {"temp_c": "mean"}

    def test_artifact_round_trip(self, tmp_path):
        study = alignment_study(semantics_matrix())
        path = tmp_path / "artifacts" / "alignment.json"
        artifact = write_alignment(study, path)
        assert path.exists()
        assert load_recommended(path) == artifact["recommended"]

    def test_load_missing_is_empty(self, tmp_path):
        assert load_recommended(tmp_path / "nope.json") == {}
