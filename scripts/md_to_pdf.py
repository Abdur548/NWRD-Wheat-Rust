"""Renders the documents in docs/ to PDFs in pdf/ (not versioned; regenerate when docs change).

    pip install -e ".[docs]"
    python scripts/md_to_pdf.py                    # every docs/*.md
    python scripts/md_to_pdf.py docs/results.md    # just one

Markdown becomes styled HTML, then a headless Chromium (Edge or Chrome) prints it. Mermaid
diagrams render with a pinned mermaid bundle that is checked against its SHA-256 before use;
offline, or on a mismatch, diagrams print as their source text. Images are rewritten to absolute
file:// URLs so they resolve from the temporary build directory.
"""

import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(ROOT, 'docs')
PDF_DIR = os.path.join(ROOT, 'pdf')
MERMAID_URL = 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js'
MERMAID_SHA256 = '61b335a46df05a7ce1c98378f60e5f3e77a7fb608a1056997e8a649304a936d6'

BROWSERS = [
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
]

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.45; color: #17202a; }
h1 { font-size: 20pt; border-bottom: 2px solid #1d6a3e; padding-bottom: 4px; color: #14532d; }
h2 { font-size: 14pt; margin-top: 18px; color: #14532d; border-bottom: 1px solid #d7e3da;
     padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 14px; color: #1d6a3e; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
code, pre { font-family: Consolas, "Courier New", monospace; font-size: 9pt; }
code { background: #f2f5f3; padding: 1px 3px; border-radius: 3px; }
pre { background: #f7f9f8; border: 1px solid #e0e7e3; border-radius: 4px; padding: 8px;
      white-space: pre-wrap; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 8.6pt;
        page-break-inside: auto; }
th, td { border: 1px solid #cfd9d3; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #eef4f0; font-weight: 600; }
tr { page-break-inside: avoid; }
blockquote { margin: 10px 0; padding: 6px 12px; border-left: 3px solid #1d6a3e;
             background: #f6faf7; color: #2f4538; }
img { max-width: 100%; height: auto; page-break-inside: avoid; }
em img, p img { display: block; margin: 8px auto; border: 1px solid #e0e7e3; }
.mermaid { text-align: center; page-break-inside: avoid; margin: 12px 0; }
/* Mermaid blocks are containers, not code: no code styling, and the rendered SVG is
   constrained to the printable area so it never spills across a page break. */
pre.mermaid { background: none; border: none; padding: 0; white-space: normal; }
.mermaid svg { display: block; margin: 0 auto; max-width: 100%; max-height: 185mm;
               width: auto; height: auto; }
a { color: #1d6a3e; }
hr { border: none; border-top: 1px solid #dfe6e2; margin: 16px 0; }
"""

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style>{mermaid}</head>
<body>{body}</body></html>"""

MERMAID_SCRIPT = (
    '<script src="mermaid.min.js"></script>'
    '<script>'
    'mermaid.initialize({startOnLoad:false,theme:"neutral",'
    'flowchart:{useMaxWidth:true,htmlLabels:true}});'
    'window.addEventListener("load", async () => {'
    '  await mermaid.run({querySelector:".mermaid"});'
    '  document.querySelectorAll(".mermaid svg").forEach(s => {'
    '    s.removeAttribute("width"); s.removeAttribute("height");'
    '    s.style.maxWidth="100%"; s.style.maxHeight="185mm";'
    '    s.style.width="auto"; s.style.height="auto";'
    '  });'
    '});'
    '</script>'
)


def find_browser() -> str:
    for p in BROWSERS:
        if os.path.exists(p):
            return p
    for name in ('msedge', 'chrome', 'chromium'):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit('No Chromium browser found for PDF printing (looked for Edge and Chrome).')


def mermaid_assets(build_dir: str) -> str:
    """Saves the verified mermaid bundle next to the HTML; returns the <script> block that
    renders diagrams, or '' when the bundle is unavailable or does not match its hash."""
    try:
        with urllib.request.urlopen(MERMAID_URL, timeout=30) as r:
            data = r.read()
    except OSError as e:
        print(f'  ! mermaid unavailable ({e}); diagrams will print as code')
        return ''
    if hashlib.sha256(data).hexdigest() != MERMAID_SHA256:
        print('  ! mermaid bundle does not match its pinned SHA-256; diagrams will print as code')
        return ''
    with open(os.path.join(build_dir, 'mermaid.min.js'), 'wb') as f:
        f.write(data)
    return MERMAID_SCRIPT


def to_html(md_path: str, build_dir: str, mermaid: str) -> str:
    import markdown

    with open(md_path, encoding='utf-8') as f:
        text = f.read()
    # ```mermaid fences must survive as <pre class="mermaid"> for the JS renderer.
    blocks = []

    def stash(m):
        blocks.append(m.group(1))
        return f'\n@@MERMAID{len(blocks) - 1}@@\n'

    text = re.sub(r'```mermaid\n(.*?)```', stash, text, flags=re.S)
    html = markdown.markdown(
        text, extensions=['tables', 'fenced_code', 'toc', 'sane_lists', 'attr_list', 'md_in_html']
    )
    for i, b in enumerate(blocks):
        html = html.replace(f'<p>@@MERMAID{i}@@</p>', f'<pre class="mermaid">{b}</pre>')
    # Relative images -> absolute file URLs, so they load from the temporary build directory.
    md_root = os.path.dirname(os.path.abspath(md_path))

    def abs_src(m):
        src = m.group(2)
        if src.startswith(('http:', 'https:', 'data:', 'file:')):
            return m.group(0)
        full = os.path.normpath(os.path.join(md_root, src)).replace('\\', '/')
        return f'{m.group(1)}file:///{full}{m.group(3)}'

    html = re.sub(r'(<img[^>]*src=")([^"]+)(")', abs_src, html)
    title = os.path.splitext(os.path.basename(md_path))[0]
    out = os.path.join(build_dir, title + '.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(PAGE.format(title=title, css=CSS, mermaid=mermaid if blocks else '', body=html))
    return out


def print_pdf(browser: str, html_path: str, pdf_path: str) -> bool:
    profile = tempfile.mkdtemp(prefix='mdpdf-')
    url = 'file:///' + html_path.replace('\\', '/')
    cmd = [
        browser,
        '--headless=new',
        '--disable-gpu',
        '--no-first-run',
        f'--user-data-dir={profile}',
        '--virtual-time-budget=20000',
        '--run-all-compositor-stages-before-draw',
        '--no-pdf-header-footer',
        f'--print-to-pdf={pdf_path}',
        url,
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    shutil.rmtree(profile, ignore_errors=True)
    return os.path.exists(pdf_path)


def main(paths: list[str]) -> int:
    os.makedirs(PDF_DIR, exist_ok=True)
    browser = find_browser()
    print('browser:', browser)
    build_dir = tempfile.mkdtemp(prefix='mdbuild-')
    mermaid = mermaid_assets(build_dir)
    ok = True
    for md_path in paths:
        name = os.path.splitext(os.path.basename(md_path))[0]
        pdf_path = os.path.join(PDF_DIR, name + '.pdf')
        print(f'{name}.md -> pdf/{name}.pdf')
        html = to_html(md_path, build_dir, mermaid)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        t0 = time.time()
        if print_pdf(browser, html, pdf_path):
            print(f'  {os.path.getsize(pdf_path) / 1024:.0f} KB in {time.time() - t0:.1f}s')
        else:
            print('  FAILED')
            ok = False
    shutil.rmtree(build_dir, ignore_errors=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or sorted(glob.glob(os.path.join(DOCS_DIR, '*.md')))))
