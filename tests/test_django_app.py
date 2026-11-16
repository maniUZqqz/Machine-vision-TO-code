"""Django web app: upload -> result (side by side) -> ZIP, and the REST API."""
import io
import json
import os
import zipfile

import pytest

django = pytest.importorskip("django")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    os.environ.setdefault("SECRET_KEY", "test-only")
    import django as dj
    from django.conf import settings
    dj.setup()
    from django.test.utils import setup_test_environment, teardown_test_environment
    from django.db import connection
    setup_test_environment()
    media = tmp_path_factory.mktemp("media")
    old_media = settings.MEDIA_ROOT
    settings.MEDIA_ROOT = str(media)
    settings.ALLOWED_HOSTS = ["testserver", "localhost"]
    old_name = connection.settings_dict["NAME"]
    connection.creation.create_test_db(verbosity=0, autoclobber=True)
    # keep the unit tests fast and independent of a browser
    from apps.converter import services
    services._score = lambda rgb, result: {"unavailable": "disabled in tests"}
    from django.test import Client
    yield Client()
    connection.creation.destroy_test_db(old_name, verbosity=0)
    settings.MEDIA_ROOT = old_media
    teardown_test_environment()


def _png():
    import cv2
    from tests.conftest import synthetic_ui
    ok, buf = cv2.imencode(".png", cv2.cvtColor(synthetic_ui(), cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def test_api_convert_and_download(client):
    from django.core.files.uploadedfile import SimpleUploadedFile
    r = client.post("/api/convert/", {"image": SimpleUploadedFile("ui.png", _png(), "image/png"), "mode": "responsive"})
    assert r.status_code == 200, r.content
    data = r.json()
    assert data["status"] == "completed", data
    assert "<html" in data["html"] and data["tree"]["role"] == "page"
    # result page: original + generated side by side + zip link
    page = client.get(f"/result/{data['job_id']}/")
    assert page.status_code == 200
    body = page.content.decode()
    assert "Original screenshot" in body and "/download/" in body
    z = client.get(f"/download/{data['job_id']}/")
    assert z.status_code == 200 and z["Content-Type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert "index.html" in names and "tree.json" in names and "original.png" in names


def test_api_errors(client):
    assert client.post("/api/convert/", {}).status_code == 400
    from django.core.files.uploadedfile import SimpleUploadedFile
    r = client.post("/api/convert/", {"image": SimpleUploadedFile("ui.png", _png(), "image/png"), "mode": "bogus"})
    assert r.status_code == 400 and "mode" in r.json()["error"]


def test_upload_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Exact" in r.content


def test_corrupt_upload_fails_cleanly(client):
    from django.core.files.uploadedfile import SimpleUploadedFile
    r = client.post("/api/convert/", {"image": SimpleUploadedFile("bad.png", b"not a png", "image/png")})
    # Django's ImageField is not used on the API route: the job itself must fail with a message
    assert r.status_code == 200 and r.json()["status"] == "failed" and r.json()["error"]
