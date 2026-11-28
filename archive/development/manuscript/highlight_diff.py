# -*- coding: utf-8 -*-
"""Produce a copy of the revised PDF in which every word that is new or changed relative to the first
submission is covered by a yellow highlight annotation (IEEE Access 'Highlighted PDF').

    python highlight_diff.py OLD.pdf NEW.pdf OUT.pdf
"""
import difflib
import re
import sys

import fitz


def words_of(doc):
    """(page, rect, normalized word) for every word of the document body, in reading order."""
    out = []
    for page in doc:
        for x0, y0, x1, y1, w, block, line, wno in page.get_text("words", sort=True):
            norm = re.sub(r"[^\w]", "", w.lower())
            if norm:
                out.append((page.number, fitz.Rect(x0, y0, x1, y1), norm))
    return out


def main(old_path, new_path, out_path):
    old, new = fitz.open(old_path), fitz.open(new_path)
    a = [w[2] for w in words_of(old)]
    nw = words_of(new)
    b = [w[2] for w in nw]
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pages = [new[i] for i in range(new.page_count)]  # keep page objects alive while annotating
    changed = 0
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            for page_no, rect, _ in nw[j1:j2]:
                annot = pages[page_no].add_highlight_annot(rect)
                annot.set_colors(stroke=(1, 1, 0))
                annot.update()
                changed += 1
    new.save(out_path, garbage=3, deflate=True)
    print(f"highlighted {changed} of {len(nw)} words ({100 * changed / len(nw):.0f}%) -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
