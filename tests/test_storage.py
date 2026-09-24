import os
import stat

import polars as pl
import pytest

from grounded_weather_forecast.storage import atomic_write_parquet, atomic_write_text
from grounded_weather_forecast.storage import stage_text


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode semantics")
def test_new_atomic_outputs_follow_umask(tmp_path):
    previous = os.umask(0o022)
    try:
        parquet = tmp_path / "scores.parquet"
        text = tmp_path / "state.json"
        atomic_write_parquet(pl.DataFrame({"x": [1]}), parquet)
        atomic_write_text("{}", text)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(parquet.stat().st_mode) == 0o644
    assert stat.S_IMODE(text.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode semantics")
def test_atomic_replacement_preserves_existing_mode(tmp_path):
    parquet = tmp_path / "scores.parquet"
    text = tmp_path / "state.json"
    pl.DataFrame({"x": [0]}).write_parquet(parquet)
    text.write_text("old", encoding="utf-8")
    parquet.chmod(0o640)
    text.chmod(0o660)

    atomic_write_parquet(pl.DataFrame({"x": [1]}), parquet)
    atomic_write_text("new", text)

    assert stat.S_IMODE(parquet.stat().st_mode) == 0o640
    assert stat.S_IMODE(text.stat().st_mode) == 0o660
    assert pl.read_parquet(parquet)["x"].to_list() == [1]
    assert text.read_text(encoding="utf-8") == "new"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode semantics")
def test_read_only_destination_is_replaced_and_keeps_mode(tmp_path):
    path = tmp_path / "read-only.json"
    path.write_text("old", encoding="utf-8")
    path.chmod(0o444)
    atomic_write_text("new", path)
    assert path.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_interrupted_stage_cleans_temporary_file(tmp_path, monkeypatch):
    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(type(tmp_path), "write_text", interrupted)
    with pytest.raises(KeyboardInterrupt):
        stage_text("new", tmp_path / "state.json")
    assert not list(tmp_path.glob("*.tmp"))
