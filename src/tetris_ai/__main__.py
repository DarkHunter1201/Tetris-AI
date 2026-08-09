from .paths import configure_environment


configure_environment()

from .app import run


raise SystemExit(run())
