"""The REST API: inference on uploads, input limits, and the files it serves."""

import base64
import io
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.backend.main import Settings, create_app
from nwrd.models import build_model


@pytest.fixture(scope='module')
def assets(tmp_path_factory):
    """A random-weight CANet-B4 checkpoint and the files the app serves."""
    root = tmp_path_factory.mktemp('assets')
    torch.manual_seed(0)
    model = build_model('canet_b4', pretrained=False)
    torch.save(
        {'model': model.state_dict(), 'model_name': 'canet_b4', 'seed': 0}, root / 'best_model.pth'
    )
    (root / 'results.json').write_text(json.dumps({'best_threshold': 0.42, 'final_metrics': {}}))
    (root / 'figures').mkdir()
    Image.new('RGB', (8, 8)).save(root / 'figures' / 'training_curves.png')
    return root


def _client(assets, **overrides):
    return TestClient(create_app(Settings(assets=assets, **overrides)))


def _upload(img: Image.Image, fmt='PNG', **save):
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save)
    return {'file': (f'photo.{fmt.lower()}', buf.getvalue(), f'image/{fmt.lower()}')}


def _photo(w=300, h=200):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _decode(data_url):
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(',', 1)[1])))


def test_predict_returns_full_resolution_mask_and_stats(assets):
    with _client(assets) as client:
        assert client.get('/api/health').json()['model_loaded'] is True
        r = client.post('/api/predict', files=_upload(_photo(700, 300)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body['width'], body['height'], body['patches']) == (700, 300, 2)
    assert body['threshold'] == 0.42  # the val-selected default from results.json
    assert body['total_pixels'] == 700 * 300 and 0 <= body['rust_pixels'] <= body['total_pixels']
    assert body['severity'] in ('healthy', 'early', 'moderate', 'severe')
    assert _decode(body['pred_mask_b64']).size == (700, 300)
    for key in ('original_b64', 'prob_map_b64', 'overlay_b64'):
        assert _decode(body[key]).size == (700, 300)  # below the thumbnail size: unchanged


def test_threshold_is_validated(assets):
    with _client(assets) as client:
        ok = client.post('/api/predict', files=_upload(_photo()), data={'threshold': '0.9'})
        assert ok.status_code == 200 and ok.json()['threshold'] == 0.9
        for bad in ('1.5', '-0.1', 'nan', 'inf', 'abc'):
            r = client.post('/api/predict', files=_upload(_photo()), data={'threshold': bad})
            assert r.status_code == 422, bad


def test_camera_orientation_is_applied(assets):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated 90°: the photo is shown portrait
    with _client(assets) as client:
        r = client.post('/api/predict', files=_upload(_photo(300, 200), 'JPEG', exif=exif))
    assert (r.json()['width'], r.json()['height']) == (200, 300)


def test_bad_and_oversized_uploads_are_refused(assets):
    with _client(assets, max_upload_bytes=50_000, max_pixels=100_000) as client:
        not_image = {'file': ('notes.txt', b'not an image', 'text/plain')}
        assert client.post('/api/predict', files=not_image).status_code == 400
        assert client.post('/api/predict', files=_upload(_photo(400, 300))).status_code == 413
        big = _upload(_photo(400, 300), 'JPEG', quality=10)  # small file, too many pixels
        assert len(big['file'][1]) < 50_000
        assert client.post('/api/predict', files=big).status_code == 413


def test_missing_model_is_reported_not_crashed(tmp_path):
    with _client(tmp_path) as client:
        assert client.get('/api/health').json()['model_loaded'] is False
        assert client.post('/api/predict', files=_upload(_photo())).status_code == 503
        assert client.get('/api/results').status_code == 404


def test_only_figures_are_served(assets):
    with _client(assets) as client:
        assert client.get('/api/images/training_curves.png').status_code == 200
        assert client.get('/api/results').json()['best_threshold'] == 0.42
        for name in (
            'results.json',
            '..%2Fresults.json',
            '..%2F..%2Fpyproject.toml',
            'missing.png',
        ):
            assert client.get(f'/api/images/{name}').status_code == 404, name


@pytest.mark.parametrize('origin', ['http://localhost:5173', 'http://127.0.0.1:5173'])
def test_frontend_dev_server_may_call_the_api(assets, origin):
    with _client(assets) as client:
        r = client.get('/api/results', headers={'Origin': origin})
    assert r.headers.get('access-control-allow-origin') == origin


def test_other_origins_are_not_allowed(assets):
    with _client(assets) as client:
        r = client.get('/api/results', headers={'Origin': 'http://evil.example'})
    assert 'access-control-allow-origin' not in r.headers
