#!/bin/sh

# Copyright (c) 2019 Abhishek Thakur(@abhiTronix) <abhi.una12@gmail.com>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Checks relative href/src links in built docs resolve to existing files,
# and that any `#fragment` matches an id/name on the target page.
# Usage: sh scripts/bash/check_links.sh [site_dir] [whitelist_file]
# Whitelist: one glob per line, matched against page path or link URL.

SITE_DIR="${1:-site}"
WHITELIST="${2:-$(dirname "$0")/check_links_whitelist.txt}"

python - "$SITE_DIR" "$WHITELIST" <<'EOF'
import re, os, sys, pathlib, fnmatch, functools

site = pathlib.Path(sys.argv[1]).resolve()
wl = pathlib.Path(sys.argv[2])
patterns = [l.strip() for l in wl.read_text().splitlines() if l.strip() and not l.startswith("#")] if wl.exists() else []
html, ids = {}, {}
def read(f):
    if f not in html:
        html[f] = f.read_text(encoding="utf-8", errors="ignore")
    return html[f]
def anchors(f):
    if f not in ids:
        ids[f] = set(re.findall(r'\b(?:id|name)=["\']?([^"\'\s>]+)', read(f)))
    return ids[f]
listdir = functools.lru_cache(maxsize=None)(os.listdir)
broken = []
for f in sorted(site.rglob("*.html")):
    page = f.relative_to(site).as_posix()
    if any(fnmatch.fnmatch(page, p) for p in patterns):
        continue
    for u in re.findall(r'(?:href|src)=["\']?([^"\'\s>]+)', read(f)):
        # skip absolute/external links, and bare `../`-only links (mike version banner)
        if re.match(r"^(\w+:|//|/)", u) or re.fullmatch(r"(\.\.?/?)+", u):
            continue
        if any(fnmatch.fnmatch(u, p) for p in patterns):
            continue
        p, _, frag = u.partition("#")
        p = p.split("?")[0]
        tgt = (f.parent / p).resolve() if p else f
        # exact-case match, so case-insensitive filesystems (Windows/macOS) fail like CI
        exact = not p or site in tgt.parents and all(
            part in listdir(site.joinpath(*tgt.relative_to(site).parts[:i]))
            for i, part in enumerate(tgt.relative_to(site).parts)
        )
        if tgt.is_dir():
            tgt = tgt / "index.html"
        if not (exact and tgt.is_file()):
            broken.append(f"{page}: {u}")
        # fragment must match an id/name on the target html page (zensical only checks same-page ones)
        elif frag and tgt.suffix == ".html" and frag not in anchors(tgt):
            broken.append(f"{page}: {u} (missing anchor)")
print(*broken, f"{len(broken)} broken link(s)", sep="\n")
sys.exit(1 if broken else 0)
EOF
