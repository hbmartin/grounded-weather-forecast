"""Typed factories and teardown shared by score recovery fixtures."""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

from filelock import FileLock

from grounded_weather_forecast.backtest.score_storage import SCORE_STAGE_NAMESPACE
from grounded_weather_forecast.storage import temporary_sibling


class OrphanFactory(Protocol):
    def __call__(
        self, directory: Path, name: str = "orphan", contents: bytes = b"orphan"
    ) -> Path: ...


type GuardFailureFactory = Callable[[Exception], None]


def create_orphan(
    directory: Path, name: str = "orphan", contents: bytes = b"orphan"
) -> Path:
    staged = temporary_sibling(
        directory / f"scores_{name}.parquet", namespace=SCORE_STAGE_NAMESPACE
    )
    staged.write_bytes(contents)
    return staged


def release_guards(guards: Iterable[FileLock], release: Callable[..., None]) -> None:
    """Attempt every release, including when an earlier guard fails."""
    failures: list[BaseException] = []
    for guard in guards:
        try:
            release(guard, force=True)
        except (BaseException,) as error:  # noqa: B013 -- finish all test cleanup
            failures.append(error)
    if failures:
        raise BaseExceptionGroup("score guard teardown failures", failures)
