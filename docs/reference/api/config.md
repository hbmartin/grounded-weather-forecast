# Configuration objects

`load_config` parses TOML into these frozen dataclasses, validating every key and
raising `ConfigError` with the offending key named. The narrative reference for
what each setting *does* is [Configuration](../configuration.md).

::: grounded_weather_forecast.config
