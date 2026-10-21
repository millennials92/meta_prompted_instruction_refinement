"""Cross-reference check: every 'Section N.M', 'Table N', 'Figure N' and 'Appendix X' mentioned in the prose must exist."""
import re
import sys

sys.path.insert(0, ".")
import content_blocks as C

headings, tables, figures, appendices = set(), set(), set(), set()
for b in C.BLOCKS:
    if b["type"] in ("h1", "h2", "h3"):
        m = re.match(r"^((?:\d+|[A-Z])(?:\.\d+)*)\.?\s", b["text"])
        if m:
            headings.add(m.group(1))
        m = re.match(r"^Appendix ([A-Z])\.", b["text"])
        if m:
            appendices.add(m.group(1))
    if b["type"] == "table":
        m = re.match(r"^Table ([A-Z]?\d+)\.", b["caption"])
        tables.add(m.group(1)) if m else None
    if b["type"] in ("figure", "code") and b.get("caption"):
        m = re.match(r"^Figure ([A-Z]?\.?\d+)\.", b["caption"])
        figures.add(m.group(1)) if m else None
    if b["type"] == "algo":
        m = re.match(r"^Algorithm (\d+)\.", b["title"])
        figures.add("Alg" + m.group(1)) if m else None

problems = []
for b in C.BLOCKS:
    texts = []
    if b["type"] == "p":
        texts = [b["text"]]
    elif b["type"] == "bullets":
        texts = [h + " " + t for h, t in b["items"]]
    for t in texts:
        for m in re.finditer(r"Sections? (\d+(?:\.\d+)*)", t):
            if m.group(1) not in headings:
                problems.append(("Section", m.group(1), t[:70]))
        for m in re.finditer(r"Tables? ([A-Z]?\d+)", t):
            if m.group(1) not in tables:
                problems.append(("Table", m.group(1), t[:70]))
        for m in re.finditer(r"Figures? ([A-Z]?\.?\d+)", t):
            if m.group(1) not in figures:
                problems.append(("Figure", m.group(1), t[:70]))
        for m in re.finditer(r"Appendix ([A-Z])", t):
            if m.group(1) not in appendices:
                problems.append(("Appendix", m.group(1), t[:70]))
print("headings", len(headings), "tables", sorted(tables), "figures", sorted(figures), "appendices", sorted(appendices))
for p in problems:
    print("MISSING", p)
print("OK" if not problems else f"{len(problems)} problems")
