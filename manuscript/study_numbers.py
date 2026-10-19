"""Numbers computed by mpir_study/analyze.py, loaded once so manuscript text never hand-copies a result."""
import json
import os

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "revision", "summary_numbers.json")


class _Numbers(dict):
    def __missing__(self, key):
        raise KeyError(f"summary_numbers.json has no entry '{key}'; run mpir_study/analyze.py")


N = _Numbers(json.load(open(_PATH, encoding="utf-8"))) if os.path.exists(_PATH) else _Numbers()
