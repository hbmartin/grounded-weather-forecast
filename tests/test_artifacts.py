import pytest

from omni_forecast.artifacts import ArtifactError, ArtifactStore


class TestArtifactStore:
    def test_save_and_load(self, tmp_path):
        store = ArtifactStore(root=tmp_path / "artifacts")
        state = {"coefficients": [1.0, 2.0], "tau": 3.0}
        slot = store.save(
            fingerprint="abc123",
            method_id="grounded_equal_weight",
            product="hourly",
            variable="temp_c",
            state=state,
            meta={"train_rows": 100},
        )
        assert (slot / "state.json").exists()
        assert (slot / "manifest.json").exists()
        loaded = store.load_state(
            fingerprint="abc123",
            method_id="grounded_equal_weight",
            product="hourly",
            variable="temp_c",
        )
        assert loaded == state

    def test_latest_pointer(self, tmp_path):
        store = ArtifactStore(root=tmp_path / "artifacts")
        store.save(
            fingerprint="aaa",
            method_id="gbm",
            product="hourly",
            variable="temp_c",
            state={},
        )
        store.save(
            fingerprint="bbb",
            method_id="gbm",
            product="hourly",
            variable="temp_c",
            state={},
        )
        latest = store.read_latest()
        assert latest["hourly.temp_c.gbm"]["fingerprint"] == "bbb"

    def test_missing_artifact_raises(self, tmp_path):
        store = ArtifactStore(root=tmp_path / "artifacts")
        with pytest.raises(ArtifactError, match="no artifact"):
            store.load_state(
                fingerprint="zzz",
                method_id="gbm",
                product="hourly",
                variable="temp_c",
            )

    def test_empty_latest(self, tmp_path):
        assert ArtifactStore(root=tmp_path / "nothing").read_latest() == {}

    def test_metadata_cannot_override_identity(self, tmp_path):
        store = ArtifactStore(root=tmp_path / "artifacts")
        with pytest.raises(ArtifactError, match="reserved keys"):
            store.save(
                fingerprint="real",
                method_id="gbm",
                product="hourly",
                variable="temp_c",
                state={},
                meta={"dataset_fingerprint": "forged"},
            )
