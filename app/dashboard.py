"""Streamlit dashboard for the CANet-B4 wheat-rust model: results, figures and live prediction.

    streamlit run app/dashboard.py        # after pip install -e ".[app]"

It reads the same files as the REST API (app/assets/) and predicts with the same code
(nwrd/inference.py): on native-resolution 512px patches, the way the test set was scored.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from nwrd import inference

ASSETS = Path(__file__).resolve().parent / 'assets'
CHECKPOINT = ASSETS / 'best_model.pth'
RESULTS = ASSETS / 'results.json'
FIGURES = ASSETS / 'figures'
SEVERITY = {
    'healthy': (st.success, 'No rust detected.'),
    'early': (st.info, 'Early: under 5% of the image is predicted as rust.'),
    'moderate': (st.warning, 'Moderate: 5–20% of the image is predicted as rust.'),
    'severe': (st.error, 'Severe: 20% or more of the image is predicted as rust.'),
}

CSS = """
<style>
    :root { --primary-green: #1D6A3E; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 2rem !important; }
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e0e0e0; border-radius: 0.5rem;
        padding: 1rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .hero {
        background: var(--primary-green); color: white; padding: 2.5rem; border-radius: 0.5rem;
        text-align: center; margin-bottom: 2rem;
    }
    .hero h1 { color: white; font-size: 2.6rem; font-weight: 800; margin-bottom: 0.5rem; }
    .hero p { font-size: 1.15rem; opacity: 0.9; }
    .info-card {
        background: white; padding: 1.25rem; border-radius: 0.5rem; margin-bottom: 1rem;
        border-left: 5px solid var(--primary-green); box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .info-card h4 { color: var(--primary-green); margin-bottom: 0.4rem; font-weight: 600; }
</style>
"""


@st.cache_resource
def load_model():
    if not CHECKPOINT.exists():
        return None, {}
    return inference.load_model(str(CHECKPOINT))


@st.cache_data
def load_results() -> dict | None:
    if not RESULTS.exists():
        return None
    return json.loads(RESULTS.read_text(encoding='utf-8'))


def card(title: str, text: str) -> str:
    return f'<div class="info-card"><h4>{title}</h4><p>{text}</p></div>'


def page_overview() -> None:
    st.markdown(
        '<div class="hero"><h1>🌾 Wheat Rust Segmentation</h1>'
        '<p>Pixel-level stripe-rust segmentation with CANet and an EfficientNet-B4 encoder</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    left.markdown(
        card(
            '📊 Dataset',
            'NWRD (NUST Wheat Rust Disease): 100 field photographs of wheat with stripe rust and '
            'pixel-level masks, split by photograph into train (85), validation (5) and test (10).',
        )
        + card('🧠 Encoder', 'EfficientNet-B4, pretrained on ImageNet and fine-tuned.'),
        unsafe_allow_html=True,
    )
    right.markdown(
        card(
            '🏗️ Architecture',
            'CANet: a context aggregation module with attention fusion, and an asymmetric decoder.',
        )
        + card(
            '⚖️ How it compares',
            'Under one controlled protocol, CANet-B4 is comparable in accuracy to the NWRD '
            "paper's UNet and to DeepLabV3+ R50 on 512px patches, and less accurate than UNet "
            "at the paper's 128px scale. It needs about a tenth of UNet's computation. Details: "
            '<code>docs/results.md</code>.',
        ),
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<div style="text-align:center;color:#666;"><b>Project team:</b> Syed Muhammad Abdur '
        'Rahman (467471), Abdul Hadi Sheikh (454448), Abdullah Salim Nizami (457223)<br>'
        '<b>Supervisor:</b> Dr. Imran Malik<br><b>Institution:</b> NUST SEECS — April 2026'
        '</div>',
        unsafe_allow_html=True,
    )


def page_results() -> None:
    st.title('📊 Results')
    results = load_results()
    if not results:
        st.error('`app/assets/results.json` not found.')
        return
    fm = results.get('final_metrics', {})
    st.markdown(
        f'Test split (10 held-out photographs) for the deployed checkpoint: CANet-B4 seed '
        f'**{results.get("seed")}**, epoch **{results.get("best_epoch")}**, the seed with the best '
        'validation IoU. Metrics are pooled over every test pixel; the mean over three seeds, '
        'and the comparison with the baselines, are in `docs/results.md`.'
    )
    cols = st.columns(3)
    for col, key, label in zip(
        cols, ('IoU', 'F1 / Dice', 'Precision'), ('IoU', 'Dice / F1', 'Precision'), strict=True
    ):
        col.metric(label, f'{fm.get(key, float("nan")):.3f}')
    cols = st.columns(3)
    cols[0].metric('Recall', f'{fm.get("Recall", float("nan")):.3f}')
    cols[1].metric('Specificity', f'{100 * fm.get("Specificity", float("nan")):.1f}%')
    cols[2].metric('Accuracy', f'{100 * fm.get("Accuracy", float("nan")):.1f}%')
    st.caption(
        'Background pixels dominate the test split, so accuracy and specificity are high for any '
        'model; IoU and Dice are the informative measures.'
    )
    st.divider()
    st.markdown('### 🎛️ Threshold')
    st.info(
        f'**{results.get("best_threshold", "n/a")}**: chosen to maximise IoU on the validation '
        'split, then applied unchanged to the test split.'
    )
    history = results.get('history')
    if history:
        st.markdown('### ⏳ Training history')
        df = pd.DataFrame(history)
        df.index = pd.RangeIndex(1, len(df) + 1, name='epoch')
        st.dataframe(df, use_container_width=True)
        st.caption('Validation IoU and Dice per epoch at threshold 0.5, pooled over all pixels.')


def page_figures() -> None:
    st.title('👁️ Figures')
    figures = (
        (
            '📈 Training curves',
            'training_curves.png',
            'Validation loss and IoU per epoch for every model (mean over seeds; band = range). '
            'CANet-B4 peaks early and then overfits; the checkpoint with the best validation IoU '
            'is the one kept.',
        ),
        (
            '🧮 Confusion matrix',
            'confusion_matrix.png',
            'Test pixels for the deployed checkpoint at its validation-selected threshold.',
        ),
        (
            '🖼️ Qualitative results',
            'qualitative_results.png',
            "Rust-positive test patches: the image, the annotators' mask, and each model's "
            'prediction at its own validation-selected threshold (seed 42).',
        ),
    )
    for title, name, caption in figures:
        st.markdown(f'### {title}')
        path = FIGURES / name
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.caption(caption)
        else:
            st.warning(f'`{name}` not found in app/assets/figures.')
        st.divider()


def page_predict() -> None:
    st.title('🔬 Live prediction')
    model, _ = load_model()
    if model is None:
        st.error('`app/assets/best_model.pth` not found (run `git lfs pull`).')
        return
    default = float((load_results() or {}).get('best_threshold', 0.5))
    threshold = st.sidebar.slider(
        'Detection threshold', 0.05, 0.95, default, 0.01,
        help='Lower finds more rust; higher is more conservative. Default: the threshold chosen '
        'on validation.',
    )  # fmt: skip
    show_overlay = st.sidebar.checkbox('Show red overlay', value=True)
    st.markdown(
        'Upload a field photograph of wheat. It is predicted at native resolution, 512px patch '
        'by patch, the way the test set was scored: about 1.5 s per patch on a CPU, so roughly '
        'two minutes for a 6016×4000 photograph (96 patches).'
    )
    uploaded = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])
    if not uploaded:
        st.info('Upload an image to start.')
        return
    try:
        rgb = inference.read_image(uploaded.getvalue())
    except inference.ImageError as e:
        st.error(str(e))
        return
    with st.spinner('Segmenting…'):
        probs = inference.predict(model, rgb)
    mask = probs >= threshold
    coverage = 100 * float(mask.mean())

    st.markdown('### 🖼️ Results')
    c1, c2, c3, c4 = st.columns(4)
    c1.image(inference.thumbnail(rgb), caption='Original', use_container_width=True)
    prob_map = inference.thumbnail((probs * 255).astype(np.uint8))
    c2.image(prob_map, caption='Probability map', use_container_width=True)
    c3.image(inference.thumbnail(mask.astype(np.uint8) * 255), caption='Binary mask',
             use_container_width=True)  # fmt: skip
    shown = inference.overlay(rgb, mask) if show_overlay else rgb
    c4.image(inference.thumbnail(shown), caption='Overlay', use_container_width=True)

    st.markdown('### 📊 Measurements')
    m1, m2, m3 = st.columns(3)
    m1.metric('Rust pixels', f'{int(mask.sum()):,}')
    m2.metric('Image pixels', f'{mask.size:,}')
    m3.metric('Predicted rust', f'{coverage:.2f}%')
    show, text = SEVERITY[inference.severity(coverage)]
    show(text)
    st.caption(
        'Severity bands are illustrative, not a validated agronomic scale; the share is of all '
        'image pixels, which may include soil or sky.'
    )
    st.download_button(
        '⬇️ Download the binary mask (full resolution)',
        data=inference.encode_png(mask.astype(np.uint8) * 255),
        file_name=f'mask_{Path(uploaded.name).stem}.png',
        mime='image/png',
    )


def main() -> None:
    st.set_page_config(
        page_title='NWRD wheat-rust segmentation',
        page_icon='🌾',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='text-align:center;'><h2 style='color:#1D6A3E;margin-bottom:0;'>"
        "🌾 CANet-B4</h2><p style='color:gray;font-size:0.9em;'>Wheat rust segmentation</p></div>",
        unsafe_allow_html=True,
    )
    st.navigation(
        [
            st.Page(page_overview, title='Overview', icon='🏠'),
            st.Page(page_results, title='Results', icon='📊'),
            st.Page(page_figures, title='Figures', icon='👁️'),
            st.Page(page_predict, title='Live prediction', icon='🔬'),
        ]
    ).run()


if __name__ == '__main__':  # streamlit run executes this file as __main__
    main()
