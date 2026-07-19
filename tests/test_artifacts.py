import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

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


class TestConcurrentSaves:
    """`predict` runs every 10 minutes over several variables, so concurrent
    saves into one store are the normal case."""

    def test_parallel_saves_keep_every_pointer_entry(self, tmp_path):
        """Regression: the pointer's read-modify-write was unlocked.

        Interleaved saves lost entries outright and could leave latest.json
        unparseable, which a later reclamation pass would read as "nothing is
        referenced" -- deleting every state tree.
        """
        store = ArtifactStore(tmp_path / "state")
        methods = [f"m{index}" for index in range(24)]

        def save(method_id: str) -> None:
            store.save(
                fingerprint="fp",
                method_id=method_id,
                product="hourly",
                variable="temp_c",
                state={"method": method_id},
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(save, methods))

        latest = json.loads((tmp_path / "state" / "latest.json").read_text())
        assert set(latest) == {f"hourly.temp_c.{method}" for method in methods}
        for method_id in methods:
            assert store.load_state(
                fingerprint="fp",
                method_id=method_id,
                product="hourly",
                variable="temp_c",
            ) == {"method": method_id}

    def test_a_corrupt_pointer_is_an_error_not_an_empty_map(self, tmp_path):
        """Reading it as empty would let reclamation delete every state tree."""
        store = ArtifactStore(tmp_path / "state")
        store.save(
            fingerprint="fp",
            method_id="ewa",
            product="hourly",
            variable="temp_c",
            state={},
        )
        (tmp_path / "state" / "latest.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(ArtifactError, match="corrupt artifact pointer"):
            store.read_latest()


class TestReclamationIsSafeUnderConcurrency:
    """Reclamation runs inside `save`'s lock, so it cannot delete a live tree.

    A slot is written before the pointer that names it. An unsynchronized
    reclamation pass reads `latest.json` inside that gap, concludes the other
    run's fingerprint is unreferenced, and deletes the tree it is still
    writing into — which is the normal case, not an edge case: `predict` runs
    every 10 minutes and a rebuild changes the fingerprint between two of them.
    """

    def test_concurrent_saves_do_not_reclaim_each_other(self, tmp_path):
        store = ArtifactStore(root=tmp_path / "artifacts")
        both_ready = Barrier(2, timeout=10)

        def save(fingerprint: str, variable: str):
            both_ready.wait()
            return store.save(
                fingerprint=fingerprint,
                method_id="gbm",
                product="hourly",
                variable=variable,
                state={"variable": variable},
                reclaim_unreferenced=True,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(save, "aaaa1111", "temp_c"),
                pool.submit(save, "bbbb2222", "dew_point_c"),
            ]
            slots = [future.result(timeout=10) for future in futures]

        assert all(slot.is_dir() for slot in slots), "a live slot was reclaimed"
        pointers = store.read_latest()
        assert len(pointers) == 2
        for pointer in pointers.values():
            assert store.load_state(
                fingerprint=pointer["fingerprint"],
                method_id=pointer["method_id"],
                product=pointer["product"],
                variable=pointer["variable"],
            ) == {"variable": pointer["variable"]}

    def test_reclamation_removes_only_unreferenced_trees(self, tmp_path):
        store = ArtifactStore(root=tmp_path / "artifacts")
        for fingerprint in ("old", "new"):
            store.save(
                fingerprint=fingerprint,
                method_id="gbm",
                product="hourly",
                variable="temp_c",
                state={},
                reclaim_unreferenced=True,
            )

        trees = sorted(c.name for c in (tmp_path / "artifacts").iterdir() if c.is_dir())
        assert trees == ["new"]
