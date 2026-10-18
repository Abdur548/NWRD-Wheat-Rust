"""Model definitions: output shapes, CANet at 128px, and compatibility with saved checkpoints."""

import os

import pytest
import torch
import torch.nn.functional as F

from nwrd.models import MODELS, ContextFlow, build_model, load_checkpoint

from .conftest import ROOT

APP_MODEL = os.path.join(ROOT, 'app', 'assets', 'best_model.pth')


@pytest.mark.parametrize('name', sorted(MODELS))
def test_models_output_logits_at_input_size(name):
    m = build_model(name, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 3, 128, 128)).shape == (1, 1, 128, 128)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match='unknown model'):
        build_model('canet_b5')


def test_canet_trains_and_predicts_at_128px():
    m = build_model('canet_b4', pretrained=False)
    m.train()
    out = m(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 1, 128, 128)
    out.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_pool_cap_leaves_512px_unchanged(monkeypatch):
    """Capping CANet's context pooling at the feature-map size (needed at 128px) must not
    change any output at the 512px scale the main experiment and the app use."""

    def uncapped(self, x):  # ContextFlow.forward before the cap
        h, w = x.shape[-2:]
        xd = F.avg_pool2d(x, self.scale) if self.scale > 1 else x
        f = self.decode(self.encode(xd))
        if self.scale > 1:
            return F.interpolate(f, (h, w), mode='bilinear', align_corners=False)
        return f

    m = build_model('canet_b4', pretrained=False).eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        capped = m(x)
        monkeypatch.setattr(ContextFlow, 'forward', uncapped)
        assert torch.equal(capped, m(x))


def test_checkpoints_round_trip_with_safe_loading(tmp_path):
    m = build_model('unet', pretrained=False)
    path = tmp_path / 'best.pt'
    torch.save({'model': m.state_dict(), 'epoch': 3, 'model_name': 'unet', 'seed': 1}, path)
    n = build_model('unet', pretrained=False)
    ckpt = load_checkpoint(n, str(path))
    assert ckpt['epoch'] == 3
    a, b = m.state_dict(), n.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_checkpoints_with_pickled_code_are_refused(tmp_path):
    class Payload:
        def __reduce__(self):
            return (print, ('this would run on load',))

    path = tmp_path / 'evil.pt'
    torch.save({'model': {}, 'payload': Payload()}, path)
    with pytest.raises(Exception, match='[Ww]eights only load failed'):
        load_checkpoint(build_model('unet', pretrained=False), str(path))


@pytest.mark.skipif(not os.path.exists(APP_MODEL), reason='app model not present (git lfs pull)')
def test_app_model_loads_strictly_into_canet():
    ckpt = load_checkpoint(build_model('canet_b4', pretrained=False), APP_MODEL)
    assert ckpt['model_name'] == 'canet_b4'
