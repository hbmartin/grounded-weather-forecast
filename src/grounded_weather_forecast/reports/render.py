"""Markdown rendering of report frames + compact stdout summaries."""

from pathlib import Path

import polars as pl


def _format_cell(value: object) -> str:
    match value:
        case None:
            return ""
        case float() as number:
            return f"{number:.3f}"
        case _:
            return str(value)


def frame_to_markdown(frame: pl.DataFrame) -> str:
    """A GitHub-flavored pipe table."""
    if frame.is_empty():
        return "_no data_\n"
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "|".join(" --- " for _ in frame.columns) + "|"
    lines = [header, divider]
    for row in frame.iter_rows():
        lines.append("| " + " | ".join(_format_cell(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_report(
    reports_dir: Path, name: str, title: str, sections: list[tuple[str, pl.DataFrame]]
) -> Path:
    """Write one markdown file of titled table sections; returns its path."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}\n"]
    for heading, frame in sections:
        parts.append(f"\n## {heading}\n")
        parts.append(frame_to_markdown(frame))
    path = reports_dir / f"{name}.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def print_summary(title: str, frame: pl.DataFrame, limit: int = 30) -> None:
    print(f"\n{title}")
    if frame.is_empty():
        print("  (no data)")
        return
    with pl.Config(tbl_rows=limit, tbl_cols=-1, tbl_width_chars=200):
        print(frame.head(limit))
