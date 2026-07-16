import polars as pl

from grounded_weather_forecast.reports.render import (
    frame_to_markdown,
    print_summary,
    write_markdown_report,
)


class TestMarkdown:
    def test_pipe_table(self):
        frame = pl.DataFrame({"a": [1, 2], "b": [1.23456, None]})
        text = frame_to_markdown(frame)
        lines = text.splitlines()
        assert lines[0] == "| a | b |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| 1 | 1.235 |"
        assert lines[3] == "| 2 |  |"

    def test_empty_frame(self):
        assert frame_to_markdown(pl.DataFrame()) == "_no data_\n"

    def test_write_report(self, tmp_path):
        frame = pl.DataFrame({"x": [1]})
        path = write_markdown_report(
            tmp_path / "reports", "test", "Test Title", [("Section", frame)]
        )
        text = path.read_text()
        assert "# Test Title" in text
        assert "## Section" in text
        assert "| x |" in text

    def test_print_summary(self, capsys):
        print_summary("title", pl.DataFrame({"x": [1]}))
        out = capsys.readouterr().out
        assert "title" in out
        print_summary("empty", pl.DataFrame())
        assert "(no data)" in capsys.readouterr().out
