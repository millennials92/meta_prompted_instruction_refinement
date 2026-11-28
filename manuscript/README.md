# Manuscript

The title and abstract follow the accepted IEEE Access article, [DOI 10.1109/ACCESS.2026.3757210](https://doi.org/10.1109/ACCESS.2026.3757210).

Rebuild the current manuscript and distribution files from the saved results:

```bash
python manuscript/build_latex_ieee.py
python manuscript/dump_plaintext.py
python manuscript/make_final_package.py
```

The build requires pdflatex and bibtex on PATH, the scientific Python dependencies (including SciPy), python-docx, and Pillow.
It does not run experiments or recompute results. Output files are written to submission_final/.

The latex/ bibliography and citation mapping remain inputs to the shared renderer.
Earlier manuscripts and development records are in [archive](../archive/).
