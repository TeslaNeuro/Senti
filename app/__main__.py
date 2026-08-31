"""Application entry point."""

from app.qt_bootstrap import configure_qt_environment

configure_qt_environment()

from app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
