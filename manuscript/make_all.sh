#!/bin/bash
# Rebuild every deliverable from the current results: analysis -> manuscript PDF -> response letter -> highlighted PDF.
set -e
cd "$(dirname "$0")/.."
PY=/c/Users/vinh.dq4/AppData/Local/anaconda3/envs/py313/python.exe
$PY -m mpir_study.analyze
cd manuscript
$PY check_refs.py
$PY build_latex_ieee.py
cd latex_ieee
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex > /tmp/p1.log
bibtex manuscript > /tmp/b.log || true
pdflatex -interaction=nonstopmode manuscript.tex > /tmp/p2.log
pdflatex -interaction=nonstopmode manuscript.tex > /tmp/p3.log
cd ..
$PY build_response.py
git show 54351ed:manuscript/latex_ieee/manuscript.pdf > /tmp/old_submission.pdf
mkdir -p ../submission_revision
cp latex_ieee/manuscript.pdf ../submission_revision/MPIR_revised_clean.pdf
$PY highlight_diff.py /tmp/old_submission.pdf latex_ieee/manuscript.pdf ../submission_revision/MPIR_revised_highlighted.pdf
cp response_to_reviewers.docx ../submission_revision/Response_to_Reviewers_Access-2026-44920.docx
cp latex_ieee/manuscript.tex ../submission_revision/MPIR_revised_clean.tex
cp latex_ieee/cas-refs.bib ../submission_revision/cas-refs.bib
echo "all deliverables rebuilt in submission_revision/"
