"""DJANGO_ENV=production must not start with the development fallback SECRET_KEY."""
import os
import subprocess
import sys

import pytest

pytest.importorskip("django")
pytest.importorskip("dotenv")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_settings(env_extra):
    env = {k: v for k, v in os.environ.items() if k not in ("SECRET_KEY", "DJANGO_ENV")}
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", "import config.settings as s; print(s.DEBUG)"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)


def test_production_refuses_insecure_fallback_key():
    r = _import_settings({"DJANGO_ENV": "production"})
    assert r.returncode != 0
    assert "SECRET_KEY" in r.stderr


def test_production_with_real_key_is_not_debug():
    r = _import_settings({"DJANGO_ENV": "production", "SECRET_KEY": "x" * 50})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


def test_development_default_still_starts():
    r = _import_settings({})
    assert r.returncode == 0, r.stderr
