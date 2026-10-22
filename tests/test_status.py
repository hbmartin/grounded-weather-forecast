import json

import pytest

from grounded_weather_forecast.status import (
    COMMAND_OUTCOME_PATH,
    CommandOutcome,
    read_outcome_hint,
    write_outcome_hint,
)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "bad",
        "[]",
        '{"exit_code": true, "failure_kind": "promotion_evidence_conflict"}',
        '{"exit_code": 0, "failure_kind": "promotion_evidence_conflict"}',
        '{"exit_code": 1, "failure_kind": null}',
    ],
)
def test_unverifiable_stage_hints_cannot_exempt_a_failed_evaluation(tmp_path, raw):
    path = tmp_path / "outcome.json"
    if raw is not None:
        path.write_text(raw)
    assert read_outcome_hint(path, 1) == CommandOutcome(1)


def test_stage_hint_round_trip_preserves_kind_and_process_exit(tmp_path, monkeypatch):
    path = tmp_path / "outcome.json"
    monkeypatch.setenv(COMMAND_OUTCOME_PATH, str(path))
    result = CommandOutcome(1, "promotion_evidence_invalid")
    write_outcome_hint(result)
    assert read_outcome_hint(path, 1) == result
    assert json.loads(path.read_text())["exit_code"] == 1
    assert read_outcome_hint(path, 75) == CommandOutcome(75)


def test_unwritable_hint_does_not_change_cli_outcome(tmp_path, monkeypatch):
    monkeypatch.setenv(COMMAND_OUTCOME_PATH, str(tmp_path / "absent" / "outcome.json"))
    write_outcome_hint(CommandOutcome(1, "backtest_product_failed"))


def test_cli_exports_backtest_outcome_without_changing_exit_contract(
    tmp_path, monkeypatch
):
    from conftest import write_config
    from grounded_weather_forecast import cli

    config = write_config(tmp_path)
    path = tmp_path / "child.json"
    monkeypatch.setenv(COMMAND_OUTCOME_PATH, str(path))
    monkeypatch.setattr(cli, "load_config", lambda *_a: config)
    monkeypatch.setattr(
        cli,
        "_cmd_backtest",
        lambda *_a: CommandOutcome(1, "promotion_evidence_conflict"),
    )
    assert cli.main(["backtest"]) == 1
    assert read_outcome_hint(path, 1).failure_kind == "promotion_evidence_conflict"
