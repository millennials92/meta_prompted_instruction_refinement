# -*- coding: utf-8 -*-
"""Block-building API shared by the manuscript part modules (unchanged from the first submission)."""
# ---------------------------------------------------------------------------
BLOCKS = []


def H1(text):
    BLOCKS.append({"type": "h1", "text": text})


def H2(text):
    BLOCKS.append({"type": "h2", "text": text})


def H3(text):
    BLOCKS.append({"type": "h3", "text": text})


def P(text):
    BLOCKS.append({"type": "p", "text": text})


def BULLETS(items):
    BLOCKS.append({"type": "bullets", "items": items})


def FIG(path, caption, full=True):
    BLOCKS.append({"type": "figure", "path": path, "caption": caption, "full": full})


def TABLE(caption, header, rows, full=True, note=None, colw=None):
    BLOCKS.append({"type": "table", "caption": caption, "header": header, "rows": rows,
                    "full": full, "note": note, "colw": colw})


def CODE(lines, caption=None, full=True):
    BLOCKS.append({"type": "code", "lines": lines, "caption": caption, "full": full})


def ALGO(title, require, ensure, steps, ret):
    BLOCKS.append({"type": "algo", "title": title, "require": require, "ensure": ensure,
                    "steps": steps, "ret": ret})


def APPENDIX_START():
    # Marks the boundary where appendix sections begin. LaTeX emits \appendix here (switching
    # subsequent \section headings to auto-numbered "Appendix A/B/C..."); DOCX is a no-op since
    # heading text there already carries an explicit "Appendix X." prefix.
    BLOCKS.append({"type": "appendix_start"})


