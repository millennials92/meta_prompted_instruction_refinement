# -*- coding: utf-8 -*-
"""Final-files package for IEEE Access (accepted manuscript): FINAL Article.pdf, LaTeX source zip, graphical
abstract (660 x 295 JPG, under 45 KB), a Word file with the graphical-abstract file name and caption, and the
supplementary archive reviewed with the earlier rounds.

Run after build_latex_ieee.py. Output: submission_final/.
"""
import os
import shutil
import subprocess
import zipfile

from docx import Document
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "submission_final")
PKG = os.path.join(OUT, "_main_pkg")
LATEX = os.path.join(HERE, "latex_ieee")
PREV_SUPP = os.path.join(ROOT, "submission_revision2", "upload_ready", "6_Supplementary_Material.zip")
KEEP_EXT = (".png", ".bib", ".cls", ".bst", ".sty", ".pfb", ".tfm", ".map", ".bbl", ".tex")
GA_NAME = "Graphical_Abstract.jpg"
GA_SIZE = (660, 295)
GA_MAX_BYTES = 45 * 1024
GA_CAPTION = ("The MPIR cycle studied in this paper: a prompt from an automatic optimizer is scored against a rubric, "
              "rewritten by a language model, and kept only if it beats the original on validation questions. With a "
              "1.7B refiner and a protected answer format, the rubric-guided rewrite was not shown to help.")
GA_CAPTION_MAX_WORDS = 60


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_source_package() -> None:
    shutil.rmtree(PKG, ignore_errors=True)
    os.makedirs(os.path.join(PKG, "figures"))
    for name in os.listdir(LATEX):
        if name.endswith(KEEP_EXT) and not name.startswith("cover_letter"):
            shutil.copy(os.path.join(LATEX, name), PKG)
    shutil.copy(os.path.join(HERE, "figures", "figure_3.png"), os.path.join(PKG, "figures"))
    path = os.path.join(PKG, "manuscript.tex")
    with open(path, encoding="utf-8") as fh:
        tex = fh.read().replace("../figures/", "figures/")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    for _ in range(2):
        run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"], PKG)
    run(["bibtex", "manuscript"], PKG)
    for _ in range(2):
        run(["pdflatex", "-interaction=nonstopmode", "manuscript.tex"], PKG)


def make_graphical_abstract(dest: str) -> int:
    src = Image.open(os.path.join(HERE, "figures", "figure_3.png")).convert("RGBA")
    canvas = Image.new("RGB", GA_SIZE, "white")
    scale = min((GA_SIZE[0] - 20) / src.width, (GA_SIZE[1] - 20) / src.height)
    size = (round(src.width * scale), round(src.height * scale))
    resized = src.resize(size, Image.LANCZOS)
    canvas.paste(resized, ((GA_SIZE[0] - size[0]) // 2, (GA_SIZE[1] - size[1]) // 2), resized)
    for quality in range(95, 20, -5):
        canvas.save(dest, "JPEG", quality=quality, optimize=True)
        if os.path.getsize(dest) < GA_MAX_BYTES:
            return quality
    raise RuntimeError("graphical abstract cannot be made smaller than 45 KB")


def make_ga_word_file(dest: str) -> None:
    words = len(GA_CAPTION.split())
    if words > GA_CAPTION_MAX_WORDS:
        raise ValueError(f"caption has {words} words, limit {GA_CAPTION_MAX_WORDS}")
    doc = Document()
    doc.add_paragraph(f"Graphical abstract file name: {GA_NAME}")
    doc.add_paragraph("Overlay file name: not applicable (no video was submitted)")
    doc.add_paragraph(f"Graphical abstract caption ({words} words): {GA_CAPTION}")
    doc.save(dest)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    build_source_package()
    shutil.copy(os.path.join(PKG, "manuscript.pdf"), os.path.join(OUT, "FINAL Article.pdf"))
    with zipfile.ZipFile(os.path.join(OUT, "LaTeX_source.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(PKG):
            for name in files:
                if name.endswith((".pdf", ".aux", ".log", ".out", ".blg", ".toc")):
                    continue
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, PKG))
    quality = make_graphical_abstract(os.path.join(OUT, GA_NAME))
    make_ga_word_file(os.path.join(OUT, "Graphical_Abstract_Info.docx"))
    if os.path.exists(PREV_SUPP):
        shutil.copy(PREV_SUPP, os.path.join(OUT, "Supplementary_Material.zip"))
    print(f"GA jpeg quality {quality}, {os.path.getsize(os.path.join(OUT, GA_NAME))} bytes")
    for name in sorted(os.listdir(OUT)):
        if name != "_main_pkg":
            print(name, os.path.getsize(os.path.join(OUT, name)))


if __name__ == "__main__":
    main()
