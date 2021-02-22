
from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix="Arvato",
    settings_files=['settings.toml', '.secrets.toml'],
    validators=[
        Validator('LOG_LEVEL',
                  must_exist=True,
                  default='INFO',
                  is_in={'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'})]
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load this files in the order.
