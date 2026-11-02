from mpir_study.data import Cell, load_p0
from mpir_study.mpir import RunSpec, protect_candidate, start_prompt
from mpir_study.templates import ANSWER_FORMAT, EXEMPLAR_HEADER


def test_protect_adds_format_and_exemplar_once():
    out = protect_candidate("Think carefully.", "hyperbaton")
    assert out.count(ANSWER_FORMAT) == 1 and out.count(EXEMPLAR_HEADER) == 1
    assert protect_candidate(out, "hyperbaton") == out.strip()


def test_start_prompt_extends_p0_only_for_v2f():
    cell = Cell("hyperbaton", "protegi", 42)
    assert start_prompt(RunSpec(cell, "v2")) == load_p0(cell)
    assert EXEMPLAR_HEADER in start_prompt(RunSpec(cell, "v2f"))
