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


def test_v3f_starts_from_exemplar_prompt_and_protects_candidates():
    cell = Cell("hyperbaton", "protegi", 42)
    assert EXEMPLAR_HEADER in start_prompt(RunSpec(cell, "v3f"))
    from mpir_study.mpir import _classify
    spec = RunSpec(cell, "v3f")
    outcome, prompt = _classify(spec, "Be careful and think.", start_prompt(spec), load_p0(cell), set())
    assert outcome == "validated" and ANSWER_FORMAT in prompt and EXEMPLAR_HEADER in prompt
