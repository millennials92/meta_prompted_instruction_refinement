# -*- coding: utf-8 -*-
"""Collect the files for the IEEE Access resubmission portal into submission_revision/upload_ready/.

Run after manuscript/make_all.sh. The main-manuscript archive is built from /tmp/pkg/main, a self-contained copy
of manuscript/latex_ieee (class, bibliography, figure and compiled PDF) that was test-compiled on its own.
"""
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
U = "submission_revision/upload_ready"
os.makedirs(U, exist_ok=True)
MAIN = "submission_revision/_main_pkg"

with zipfile.ZipFile(f"{U}/1a_Main_Manuscript_LaTeX_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(MAIN):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, MAIN))

shutil.copy("submission_revision/Response_to_Reviewers_Access-2026-44920.docx", f"{U}/2_Author_Response.docx")
shutil.copy("submission_revision/MPIR_revised_highlighted.pdf", f"{U}/4_Highlighted_Manuscript_of_Changes.pdf")
shutil.copy("submission_revision/Cover_Letter.pdf", f"{U}/5_Cover_Letter.pdf")

with zipfile.ZipFile(f"{U}/6_Supplementary_Material.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in ("REVISION_PROTOCOL.md", "REBUILD.md", "README.md"):
        if os.path.exists(p):
            z.write(p, f"supplementary/{p}")
    for d in ("mpir_study", "results/revision/rounds", "results/revision/gen_cache"):
        for root, _, files in os.walk(d):
            if "__pycache__" in root:
                continue
            for f in files:
                p = os.path.join(root, f)
                z.write(p, "supplementary/" + p.replace(os.sep, "/"))
    for f in os.listdir("results/revision"):
        p = os.path.join("results/revision", f)
        if os.path.isfile(p) and f.endswith(".json") and "backup" not in f:
            z.write(p, "supplementary/results/revision/" + f)

for f in sorted(os.listdir(U)):
    print(f, round(os.path.getsize(f"{U}/{f}") / 1e6, 1), "MB")
