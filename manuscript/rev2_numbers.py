"""Numbers of the second revision (mpir_study/revision2.py, revision2_new.py, lenient_validation.py).

Loaded once so manuscript text never hand-copies a result. Missing files give empty dicts, so the manuscript still
builds while experiments are running."""
import json
import os

_REV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "revision")


def _load(name: str) -> dict:
    path = os.path.join(_REV, name)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


OFF = _load("rev2_offline.json")
NEW = _load("rev2_new.json")
AUDIT = _load("rev2_audit_labels.json")
LEN = _load("lenient_validation.json")
MID = _load("rev2_mid_refiner.json")
