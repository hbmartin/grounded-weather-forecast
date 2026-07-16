# SQLite files are the only upstream contract

grounded-weather-forecast consumes two SQLite databases — the aw2sqlite `observations` file and the
omni-weather-forecast-apis archive — as its entire interface to the upstream projects. We
deliberately do not import either upstream Python package and there is no fetch-fresh path
in `predict`; fresh data arrives only via the upstream cron. This keeps the harness
runnable against any archive vintage (backtests must work on years-old files), immune to
upstream release churn, and honest about the append-only nature of the archive. The cost —
no shared Pydantic models, schema drift must be tolerated defensively — was accepted
knowingly; the schemas are documented in `aw2sqlite-database.md` and
`omni-weather-forecast-apis-database.md`.
