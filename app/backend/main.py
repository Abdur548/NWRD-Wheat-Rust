"""REST API for the CANet-B4 model: rust segmentation of whole wheat photographs.

    uvicorn app.backend.main:app        # from the repository root; serves http://localhost:8000

The model (app/assets/best_model.pth) is the CANet-B4 seed the main experiment chose on
validation. A photograph is predicted on native-resolution 512px patches, exactly as the test set
was scored (nwrd/inference.py). Settings come from environment variables:

    NWRD_ASSETS          folder holding best_model.pth, results.json and figures/ (app/assets)
    NWRD_CORS_ORIGINS    comma-separated origins allowed to call the API (the frontend's dev
                         server: http://localhost:5173,http://127.0.0.1:5173)
    NWRD_MAX_UPLOAD_MB   largest accepted upload (25)
    NWRD_MAX_PIXELS      largest accepted image, in pixels (40000000)
"""

import base64
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import nwrd
from nwrd import inference

log = logging.getLogger('nwrd.api')
ASSETS = Path(__file__).resolve().parents[1] / 'assets'
# The frontend's dev server, under both names browsers use for this machine.
DEV_ORIGINS = ('http://localhost:5173', 'http://127.0.0.1:5173')


@dataclass(frozen=True)
class Settings:
    assets: Path = ASSETS
    cors_origins: tuple[str, ...] = DEV_ORIGINS
    max_upload_bytes: int = 25 * 2**20
    max_pixels: int = inference.MAX_PIXELS

    @classmethod
    def from_env(cls) -> 'Settings':
        origins = os.environ.get('NWRD_CORS_ORIGINS', ','.join(DEV_ORIGINS))
        return cls(
            assets=Path(os.environ.get('NWRD_ASSETS', ASSETS)),
            cors_origins=tuple(o.strip() for o in origins.split(',') if o.strip()),
            max_upload_bytes=int(float(os.environ.get('NWRD_MAX_UPLOAD_MB', 25)) * 2**20),
            max_pixels=int(os.environ.get('NWRD_MAX_PIXELS', inference.MAX_PIXELS)),
        )


def _png_data_url(array: np.ndarray) -> str:
    return 'data:image/png;base64,' + base64.b64encode(inference.encode_png(array)).decode('ascii')


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    checkpoint = settings.assets / 'best_model.pth'
    results_path = settings.assets / 'results.json'
    figures = settings.assets / 'figures'

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model, app.state.model_info = None, {}
        if checkpoint.exists():
            app.state.model, app.state.model_info = inference.load_model(str(checkpoint))
            log.info('model loaded from %s', checkpoint)
        else:
            log.error('no model at %s (git lfs pull); /api/predict will return 503', checkpoint)
        yield

    app = FastAPI(
        title='NWRD wheat-rust segmentation API', version=nwrd.__version__, lifespan=lifespan
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=['GET', 'POST'],
        allow_headers=['*'],
    )

    def read_results() -> dict | None:
        if not results_path.exists():
            return None
        return json.loads(results_path.read_text(encoding='utf-8'))

    @app.get('/api/health')
    def health() -> dict:
        model = app.state.model
        return {
            'status': 'ok',
            'model_loaded': model is not None,
            'model': app.state.model_info.get('model_name'),
            'device': str(next(model.parameters()).device) if model is not None else None,
        }

    @app.get('/api/results')
    def get_results() -> dict:
        results = read_results()
        if results is None:
            raise HTTPException(status_code=404, detail='results.json not found')
        return results

    @app.get('/api/images/{name}')
    def get_image(name: str) -> FileResponse:
        path = figures / Path(name).name  # a bare file name: no path traversal
        if path.suffix != '.png' or not path.is_file():
            raise HTTPException(status_code=404, detail='Image not found')
        return FileResponse(path)

    @app.post('/api/predict')
    def predict(
        file: Annotated[UploadFile, File(description='a photograph (JPEG, PNG or TIFF)')],
        threshold: Annotated[
            float | None,
            Form(ge=0.0, le=1.0, allow_inf_nan=False, description='default: the val-selected one'),
        ] = None,
    ) -> dict:
        # A plain `def` endpoint runs in FastAPI's threadpool, so inference never blocks the
        # event loop.
        if app.state.model is None:
            raise HTTPException(status_code=503, detail='Model not loaded (git lfs pull)')
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            limit = settings.max_upload_bytes / 2**20
            raise HTTPException(status_code=413, detail=f'Upload larger than {limit:g} MB')
        try:
            rgb = inference.read_image(data, settings.max_pixels)
        except inference.ImageTooLarge as e:
            raise HTTPException(status_code=413, detail=str(e)) from None
        except inference.ImageError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        if threshold is None:
            threshold = float((read_results() or {}).get('best_threshold', 0.5))

        t0 = time.perf_counter()
        probs = inference.predict(app.state.model, rgb)
        seconds = time.perf_counter() - t0
        mask = probs >= threshold
        coverage = 100 * float(mask.mean())
        h, w = mask.shape
        return {
            'width': w,
            'height': h,
            'patches': math.ceil(h / inference.PATCH) * math.ceil(w / inference.PATCH),
            'seconds': round(seconds, 2),
            'threshold': threshold,
            'rust_pixels': int(mask.sum()),
            'total_pixels': int(mask.size),
            'coverage_pct': round(coverage, 2),
            'severity': inference.severity(coverage),
            # Display images are downscaled; the mask is full resolution, for download.
            'original_b64': _png_data_url(inference.thumbnail(rgb)),
            'prob_map_b64': _png_data_url(inference.thumbnail((probs * 255).astype(np.uint8))),
            'overlay_b64': _png_data_url(inference.thumbnail(inference.overlay(rgb, mask))),
            'pred_mask_b64': _png_data_url(mask.astype(np.uint8) * 255),
        }

    return app


app = create_app()
