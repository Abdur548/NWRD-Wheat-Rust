"""The Streamlit dashboard: every page renders without errors, using Streamlit's test harness."""

import sys

import pytest
from streamlit.testing.v1 import AppTest

from .conftest import ROOT


@pytest.fixture(autouse=True)
def _restore_main_module():
    """AppTest leaves sys.modules['__main__'] pointing at its temporary script. Processes spawned
    later in the same session (DataLoader workers on Windows) re-import __main__ and would hang."""
    main = sys.modules['__main__']
    yield
    sys.modules['__main__'] = main


def test_dashboard_starts_on_the_overview():
    at = AppTest.from_file(f'{ROOT}/app/dashboard.py', default_timeout=120).run()
    assert not at.exception
    text = ' '.join(m.value for m in at.markdown)
    assert 'NWRD (NUST Wheat Rust Disease)' in text and 'docs/results.md' in text


def _results():
    from app.dashboard import page_results

    page_results()


def _figures():
    from app.dashboard import page_figures

    page_figures()


def _predict():
    from app.dashboard import page_predict

    page_predict()


@pytest.mark.parametrize('page', [_results, _figures, _predict])
def test_page_renders(page):
    at = AppTest.from_function(page, default_timeout=120).run()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]  # assets missing or unreadable
