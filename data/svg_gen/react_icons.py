"""
Pull the Font Awesome icon pack ("fa") from react-icons (react-icons.github.io
aggregates several icon sets; each is published as a generated JS module with
icons expressed as a JSON tag/attr/child tree -- e.g. FaHeart, FaHome, ...).
Fetched directly from the jsDelivr npm CDN (no need to download the whole
~22MB package), parsed with a regex (the file is machine-generated and has a
completely regular shape), converted to real SVG markup, and rasterized to a
small grayscale bitmap via svglib+reportlab (pure Python, no native deps).
"""
import os
import re
import json
import urllib.request
import numpy as np
from io import BytesIO
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image

CDN_URL = "https://cdn.jsdelivr.net/npm/react-icons@5.7.0/fa/index.mjs"
CACHE_PATH = os.path.join(os.path.dirname(__file__), 'raw', 'fa_index.mjs')

def _fetch_source():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    if not os.path.exists(CACHE_PATH):
        with urllib.request.urlopen(CDN_URL) as resp:
            text = resp.read().decode('utf-8')
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            f.write(text)
    with open(CACHE_PATH, encoding='utf-8') as f:
        return f.read()

def load_icons():
    """Returns dict: icon_name (e.g. 'FaHeart') -> json tree."""
    text = _fetch_source()
    matches = re.findall(r'export function (\w+) \(props\) \{\s*return GenIcon\((\{.*?\})\)\(props\);\s*\}', text)
    return {name: json.loads(tree_str) for name, tree_str in matches}

def humanize(icon_name):
    """'FaYCombinator' -> 'Y Combinator', 'FaHome' -> 'Home'."""
    name = icon_name[2:] if icon_name.startswith('Fa') else icon_name
    words = re.findall(r'[A-Z][a-z0-9]*|[A-Z]+(?=[A-Z]|$)', name)
    return ' '.join(words) if words else name

def tree_to_svg(tree, root=True):
    tag = tree['tag']
    attr = dict(tree.get('attr', {}))
    if root:
        attr['xmlns'] = 'http://www.w3.org/2000/svg'
        attr.setdefault('fill', 'black')
    attr_str = ' '.join(f'{k}="{v}"' for k, v in attr.items())
    children = tree.get('child', [])
    inner = ''.join(tree_to_svg(c, root=False) for c in children)
    return f'<{tag} {attr_str}>{inner}</{tag}>' if inner or tag == 'svg' else f'<{tag} {attr_str}/>'

def rasterize(tree, size=28, render_size=128):
    svg_str = tree_to_svg(tree)
    drawing = svg2rlg(BytesIO(svg_str.encode('utf-8')))
    buf = BytesIO()
    # render larger then downsize for antialiasing, matching typical icon-font style
    scale = render_size / max(drawing.width, drawing.height)
    drawing.width *= scale
    drawing.height *= scale
    drawing.scale(scale, scale)
    renderPM.drawToFile(drawing, buf, fmt='PNG', bg=0xffffff)
    buf.seek(0)
    img = Image.open(buf).convert('L').resize((size, size), Image.LANCZOS)
    arr = np.array(img)
    return (255 - arr).astype(np.uint8)  # invert: white glyph on black, like MNIST


if __name__ == '__main__':
    icons = load_icons()
    print(f"{len(icons)} icons loaded")
    for name in list(icons)[:5]:
        print(name, '->', humanize(name))
    arr = rasterize(icons['FaHeart'])
    print('rasterized shape', arr.shape, 'range', arr.min(), arr.max())
