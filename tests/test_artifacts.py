import json

import pytest

from grounded_weather_forecast.artifacts import ArtifactError, ArtifactStore


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
        fingerprint, state = store.load_latest_state(
            method_id="gbm", product="hourly", variable="temp_c"
        )
        assert fingerprint == "bbb"
        assert state == {}

    def test_latest_state_rejects_inconsistent_pointer(self, tmp_path):
        store = ArtifactStore(root=tmp_path / "artifacts")
        store.save(
            fingerprint="aaa",
            method_id="gbm",
            product="hourly",
            variable="temp_c",
            state={},
        )
        latest = store.read_latest()
        latest["hourly.temp_c.gbm"]["variable"] = "wind_speed_ms"
        store._latest_path().write_text(json.dumps(latest), encoding="utf-8")
        with pytest.raises(ArtifactError, match="inconsistent"):
            store.load_latest_state(
                method_id="gbm", product="hourly", variable="temp_c"
            )

    def test_missing_artifact_raises(self, tmp_path):
        store = ArtifactStore(root=tmp_path / "artifacts")
        with pytest.raises(ArtifactError, match="no artifact"):
            store.load_state(
                fingerprint="zzz",
                method_id="gbm",
                product="hourly",
                variable="temp_c",
            )

    @pytest.mark.parametrize(
        ("filename", "loader_name", "kind"),
        [
            ("state.json", "load_state", "state"),
            ("manifest.json", "load_manifest", "manifest"),
        ],
    )
    @pytest.mark.parametrize("payload", [b"{", b"\xff"])
    def test_corrupt_json_is_wrapped(
        self, tmp_path, filename, loader_name, kind, payload
    ):
        store = ArtifactStore(root=tmp_path / "artifacts")
        slot = store.save(
            fingerprint="abc123",
            method_id="gbm",
            product="hourly",
            variable="temp_c",
            state={},
        )
        (slot / filename).write_bytes(payload)

        loader = getattr(store, loader_name)
        with pytest.raises(ArtifactError, match=f"corrupt artifact {kind}"):
            loader(
                fingerprint="abc123",
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
