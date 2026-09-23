import os
import stat

import polars as pl
import pytest

from grounded_weather_forecast.storage import atomic_write_parquet, atomic_write_text


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
