# -*- coding: utf-8 -*-
"""Appendix G (second revision): the kind of every base prompt, so readers can judge how informative a cell is."""
from blocks_api import H1, P, TABLE  # noqa: F401
from rev2_numbers import OFF

H1("Appendix G. The 45 Base Prompts")
_BP = OFF["base_prompt_inventory"]
_KIND = {"seed_plus_format": "seed instruction + format line", "induction_template": "APE induction template",
         "promptwizard_long": "PromptWizard prompt with worked examples", "short_instruction": "short instruction"}
P("Table 17 lists the kind of every base prompt so that readers can judge how informative a cell is. "
  f"{_BP['kinds'].get('seed_plus_format', 0)} of the {_BP['n']} prompts are the seed instruction plus the "
  f"answer-format line (no optimization is visible in them), {_BP['kinds'].get('induction_template', 0)} are APE's "
  f"own induction template, {_BP['kinds'].get('short_instruction', 0)} are short instructions and "
  f"{_BP['kinds'].get('promptwizard_long', 0)} are long PromptWizard prompts with worked examples. A refinement "
  "layer therefore had something substantial to improve in at most the PromptWizard cells and a handful of others.")
_rows = []
for _k, _v in _BP["cells"].items():
    _t, _o, _s = _k.split("__")
    _rows.append([_t, _o, _s[1:], str(_v["chars"]), _KIND[_v["kind"]], "yes" if _v["has_bare_tag_example"] else "no"])
TABLE("Table 17. Kind of each base prompt P0 (Qwen3-1.7B-driven optimizers). Chars is the prompt length; bare-tag "
      "example means that the prompt contains a worked example whose answer is a bare option inside the answer tags.",
      header=["Task", "Optimizer", "Split seed", "Chars", "Kind", "Bare-tag example"],
      rows=_rows, full=True, colw=[0.26, 0.14, 0.09, 0.09, 0.30, 0.12])
