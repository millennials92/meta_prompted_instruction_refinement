# -*- coding: utf-8 -*-
"""Clean plain-text dump of the manuscript, in reading order, with citations resolved to
(Author, Year) form -- for feeding to review tooling. Reuses build_latex.py's citation
resolution against the same cas-refs.bib the real build uses, but emits plain text/Markdown
instead of LaTeX markup.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import content_blocks as C
import build_latex as L

CITE_RE = re.compile(r"‹([\d,]+)›")

# Build reference_N -> "Author, Year" from the actual .bib file used by the real build.
BIB_TEXT = open(os.path.join(HERE, "latex", "cas-refs.bib"), encoding="utf-8").read()
BIB_ENTRIES = {}
for m in re.finditer(r"@\w+\{(\w+),(.*?)\n\}", BIB_TEXT, re.S):
    key, body = m.groups()
    author_m = re.search(r"author\s*=\s*\{(.*?)\},?\s*\n", body, re.S)
    year_m = re.search(r"year\s*=\s*\{(.*?)\}", body)
    title_m = re.search(r"title\s*=\s*\{(.*?)\},?\s*\n", body, re.S)
    authors = author_m.group(1) if author_m else "?"
    authors = re.sub(r"\s+", " ", authors).strip()
    first_author = authors.split(" and ")[0]
    surname = re.sub(r"[{}]", "", first_author.split(",")[0]).strip()
    year = year_m.group(1) if year_m else "?"
    title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else "?"
    n_authors = len(authors.split(" and "))
    tag = surname if n_authors == 1 else (f"{surname} et al." if n_authors > 2 else
          f"{surname} & {re.sub(r'[{}]', '', authors.split(' and ')[1].split(',')[0]).strip()}")
    BIB_ENTRIES[key] = (f"{tag}, {year}", title)


def resolve(text):
    def _sub(m):
        keys = [L.REFKEY_TO_BIBKEY[f"reference_{n.strip()}"] for n in m.group(1).split(",")]
        tags = [BIB_ENTRIES.get(k, (k, "?"))[0] for k in keys]
        return "(" + "; ".join(tags) + ")"
    return CITE_RE.sub(_sub, text)


out = [
    f"# TITLE\n{C.TITLE}\n",
    f"# AUTHORS\n" + ", ".join(a["name"] for a in C.AUTHORS) + "\n",
    f"# KEYWORDS\n{C.KEYWORDS}\n",
    f"# ABSTRACT\n{resolve(C.ABSTRACT)}\n",
]
for b in C.BLOCKS:
    t = b["type"]
    if t == "appendix_start":
        out.append("\n=== APPENDIX BEGINS ===\n")
    elif t == "h1":
        out.append(f"\n# {b['text']}\n")
    elif t == "h2":
        out.append(f"\n## {b['text']}\n")
    elif t == "h3":
        out.append(f"\n### {b['text']}\n")
    elif t == "p":
        out.append(resolve(b["text"]))
    elif t == "bullets":
        for head, body in b["items"]:
            out.append(f"  - {head} {resolve(body)}")
    elif t == "figure":
        out.append(f"[FIGURE: {b['path']}] Caption: {resolve(b['caption'])}")
    elif t == "code":
        out.append("[CODE BLOCK]")
        out.append("\n".join(b["lines"]))
        if b.get("caption"):
            out.append(f"Caption: {resolve(b['caption'])}")
    elif t == "algo":
        out.append(f"[ALGORITHM: {b['title']}]")
        out.append(f"Require: {resolve(b['require'])}")
        out.append(f"Ensure: {resolve(b['ensure'])}")
        for s in b["steps"]:
            out.append(f"  {s}")
        out.append(f"Return: {b['ret']}")
    elif t == "table":
        out.append(f"[TABLE: {resolve(b['caption'])}]")
        out.append(" | ".join(b["header"]))
        for row in b["rows"]:
            out.append(" | ".join(row))
    out.append("")

body_text = "\n".join(out)

# References section
refs = []
seen = set()
for m in CITE_RE.finditer(body_text):
    pass  # citations already resolved to author/year inline above; build a full ref list separately

ref_lines = ["\n# References\n"]
for ref_key in sorted(L.REFKEY_TO_BIBKEY, key=lambda k: int(k.split("_")[1])):
    bibkey = L.REFKEY_TO_BIBKEY[ref_key]
    tag, title = BIB_ENTRIES.get(bibkey, (bibkey, "?"))
    ref_lines.append(f"- {tag}. {title}.")

full_text = body_text + "\n".join(ref_lines)
out_path = os.path.join(HERE, "manuscript_plaintext.txt")
open(out_path, "w", encoding="utf-8").write(full_text)
print("wrote", out_path, len(full_text), "chars", len(full_text.split()), "words")
