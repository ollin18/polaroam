"""MkDocs hook: add a content hash to figure URLs so browsers never show stale images.

``assets/figures/x.png#only-light`` becomes ``assets/figures/x.png?v=1a2b3c4d#only-light``.
The hash changes only when the file does, so unchanged figures stay cached.
"""

import functools
import hashlib
import re
from pathlib import Path

_FIG = re.compile(r'(?P<url>(?:\.\./)*assets/figures/(?P<name>[\w.-]+\.png))(?P<rest>[#"\'])')


@functools.cache
def _digest(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:8]


def on_page_content(html, page, config, files, **kwargs):
    figures = Path(config["docs_dir"]) / "assets" / "figures"

    def stamp(m: re.Match) -> str:
        f = figures / m["name"]
        return f"{m['url']}?v={_digest(str(f))}{m['rest']}" if f.exists() else m[0]

    return _FIG.sub(stamp, html)
