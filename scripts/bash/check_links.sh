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

# Checks relative href/src links in built docs resolve to existing files.
# Usage: sh scripts/bash/check_links.sh [site_dir] [whitelist_file]
# Whitelist: one glob per line, matched against page path or link URL.

SITE_DIR="${1:-site}"
WHITELIST="${2:-$(dirname "$0")/check_links_whitelist.txt}"

python - "$SITE_DIR" "$WHITELIST" <<'EOF'
import re, os, sys, pathlib, fnmatch

site = pathlib.Path(sys.argv[1]).resolve()
wl = pathlib.Path(sys.argv[2])
patterns = [l.strip() for l in wl.read_text().splitlines() if l.strip() and not l.startswith("#")] if wl.exists() else []
broken = []
for f in sorted(site.rglob("*.html")):
    page = f.relative_to(site).as_posix()
    if any(fnmatch.fnmatch(page, p) for p in patterns):
        continue
    for u in re.findall(r'(?:href|src)=["\']?([^"\'\s>]+)', f.read_text(encoding="utf-8", errors="ignore")):
        # skip absolute/external/anchors, and bare `../`-only links (mike version banner)
        if re.match(r"^(\w+:|//|#|/)", u) or re.fullmatch(r"(\.\.?/?)+", u):
            continue
        if any(fnmatch.fnmatch(u, p) for p in patterns):
            continue
        p = u.split("#")[0].split("?")[0]
        tgt = (f.parent / p).resolve()
        # exact-case match, so case-insensitive filesystems (Windows/macOS) fail like CI
        exact = site in tgt.parents and all(
            part in os.listdir(site.joinpath(*tgt.relative_to(site).parts[:i]))
            for i, part in enumerate(tgt.relative_to(site).parts)
        )
        if not (exact and (tgt.is_file() or (tgt / "index.html").is_file())):
            broken.append(f"{page}: {u}")
print(*broken, f"{len(broken)} broken link(s)", sep="\n")
sys.exit(1 if broken else 0)
EOF
