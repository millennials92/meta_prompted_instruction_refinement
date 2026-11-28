# Reference audit (MPIR manuscript, IEEE Access revision)

Audit date 2026-10-20. Sources used: arXiv API, Crossref API, ICLR/JMLR/MLSys pages, publisher pages. Items I could not confirm are marked UNVERIFIED.
No existing file was modified. `reference_N` keys refer to `refkey_to_bibkey.json`; printed numbers were derived from first-citation order of `\cite{}` calls in `latex_ieee/manuscript.tex` (IEEEtran, numeric, order of first citation).

## 1. Mapping used: printed [N] -> bib key -> short title

| [N] | bib key | short title |
|---|---|---|
| 1 | goyal2023 | News summarization and evaluation in the era of GPT-3 |
| 2 | qin2025 | LLMs meet NLP: a survey |
| 3 | zhang2023 | Extractive summarization via ChatGPT |
| 4 | colombo2024 | SaulLM-7B |
| 5 | zhang2025 | AlpaCare |
| 6 | errica2025 | What did I do wrong? (prompt sensitivity) |
| 7 | liu2023 | Pre-train, prompt, predict survey |
| 8 | wei2022 | Chain-of-thought prompting |
| 9 | kong2024 | Role-play prompting |
| 10 | brown2020 | GPT-3 few-shot learners |
| 11 | reynolds2021 | Prompt programming |
| 12 | zamfirescupereira2023 | Why Johnny can't prompt |
| 13 | mayilvaghanan2025 | PROPEL |
| 14 | pryzant2023 | ProTeGi (gradient descent + beam search) |
| 15 | guo2024 | EvoPrompt |
| 16 | yang2024b | Dual-phase accelerated prompt optimization |
| 17 | winata2023 | Multilingual few-shot via LM retrieval |
| 18 | min2022 | Rethinking the role of demonstrations |
| 19 | kojima2022 | Zero-shot reasoners |
| 20 | wang2024 | CoT reasoning without prompting |
| 21 | wang2023 | Towards understanding CoT |
| 22 | wang2024b | Role prompting guided domain adaptation |
| 23 | kim2024 | Persona is a double-edged sword |
| 24 | zheng2024 | Step-back prompting |
| 25 | codaforno2024 | CogBench |
| 26 | yu2025 | Mitigate position bias (hidden-state channel) |
| 27 | liu2024 | Lost in the middle |
| 28 | wang2023b | Self-consistency |
| 29 | ramnath2025 | Survey of automatic prompt optimization |
| 30 | cui2025 | Heuristic search in APO: survey |
| 31 | zhou2023 | APE |
| 32 | yang2024 | OPRO |
| 33 | agarwal2025 | PromptWizard |
| 34 | khattab2024 | DSPy |
| 35 | yuksekgonul2025 | TextGrad |
| 36 | madaan2023 | Self-Refine |
| 37 | shinn2023 | Reflexion |
| 38 | ye2024 | PE2 |
| 39 | chakma2026 | Two-stage prompt optimization for few-shot RE |
| 40 | agrawal2026 | GEPA |
| 41 | lin2025 | Prompt reinforcing |
| 42 | nair2025 | Tournament of prompts |
| 43 | singh2026 | Error taxonomy-guided prompt optimization |
| 44 | liu2025 | Content-format integrated prompt optimization |
| 45 | li2025 | Survey of automatic prompt engineering |
| 46 | schulhoff2024 | The Prompt Report |
| 47 | sahoo2024 | Systematic survey of prompt engineering |
| 48 | boonstra2025 | Kaggle/Google Prompt Engineering whitepaper |
| 49 | anthropic2025 | Anthropic prompt engineering overview |
| 50 | openai2025 | OpenAI prompt engineering best practices |
| 51 | microsoft2025 | Microsoft prompt engineering techniques |
| 52 | suzgun2023 | BIG-Bench Hard |
| 53 | qwenteam2025 | Qwen3 technical report |

The mapping agrees with every reviewer-quoted number ([1] news summarization, [3] extractive ChatGPT, [17] Winata, [22] role-prompting domain adaptation, [25] CogBench, [26]/[27] position bias, [23] Kim, [48]-[51] industry, [53] Qwen3).
Bib entries NOT cited in `manuscript.tex` (so absent from the printed list): `reynolds2021b` (exact duplicate of `reynolds2021`; delete), `cobbe2021`, `demar2006`, `dror2018`, `card2020`, `mizrahi2024`, `openai2025b`, `openai2025c`, `googledeepmind2026`, `openai2026`. The four statistics papers are good candidates to cite in the new Methods/Limitations text.

## 2. Per-reference verdict

Legend: OK = bibliographic facts verified and use is appropriate; FIX = metadata or wording change; REPLACE = swap the source.

| [N] | key | verdict | problem / finding |
|---|---|---|---|
| 1 | goyal2023 | REPLACE (for translation) | Paper is news summarization with GPT-3 (arXiv 2209.12356, Sept 2022). It does not evaluate translation. Year in bib (2023) is wrong; arXiv-only, no venue found. Keep it, but move it to the *summarization* slot and fix year to 2022 plus eprint. Translation source: Vilar2023 (ACL 2023, verified). |
| 2 | qin2025 | FIX | Verified arXiv 2405.12819 (submitted 2024-05); accepted in Frontiers of Computer Science, DOI 10.1007/s11704-025-50472-3 (arXiv comment). A broad survey; fine as a general citation for "LLMs across NLP tasks" but it is a weak citation for "summarization" specifically. Convert to `@article` with DOI. |
| 3 | zhang2023 | REPLACE (for QA) | Verified: Findings of EMNLP 2023, pp. 3270-3278, DOI 10.18653/v1/2023.findings-emnlp.214. It is about extractive summarization, not QA. Use Kamalloo2023 for QA. Add pages/DOI if kept anywhere. |
| 4 | colombo2024 | OK | Verified arXiv 2403.03883, authors match. |
| 5 | zhang2025 | FIX | arXiv 2310.14558; first submitted 2023-10 (Semantic Scholar: arXiv, year 2023), no published venue found. Authors in bib match arXiv. Year should be 2023 (or state "arXiv v-latest"); add `journal = {arXiv preprint arXiv:2310.14558}`. |
| 6 | errica2025 | FIX | Verified NAACL 2025 Long, pp. 1543-1558, DOI 10.18653/v1/2025.naacl-long.73. Official proceedings title now reads "Nations of the Americas Chapter"; bib says "North American Chapter" (acceptable but add DOI). Also cited for "small phrasing changes produce large output differences": supported. Complement with Sclar2024 (new). |
| 7 | liu2023 | FIX | Verified ACM Computing Surveys vol. 55, DOI 10.1145/3560815 (Crossref gives no article number; bib's "9, Article 195" is from memory of the ACM page, UNVERIFIED). Convert from `@misc` to `@article`. |
| 8 | wei2022 | OK | NeurIPS 2022 (vol. 35). Add DOI/pages optional. |
| 9 | kong2024 | OK | Verified NAACL 2024 Long, pp. 4099-4113, DOI 10.18653/v1/2024.naacl-long.228, authors match. Supports "role-play prompting improves zero-shot reasoning" (their claim). |
| 10 | brown2020 | OK | NeurIPS 2020, 33. |
| 11 | reynolds2021 | OK | Verified CHI EA 2021, DOI 10.1145/3411763.3451760, 7 pages. Delete duplicate `reynolds2021b`. |
| 12 | zamfirescupereira2023 | FIX | Verified CHI 2023, DOI 10.1145/3544548.3581388 (21 pages). Add DOI. |
| 13 | mayilvaghanan2025 | FIX | Verified, DOI 10.18653/v1/2025.knowledgenlp-1.25, pp. 272-302. Official venue string "4th International Workshop on Knowledge-Augmented Methods for NLP"; bib omits "International". Cited in Sec. I for "substantial trial-and-error" of manual prompting: PROPEL is an optimizer paper; weak support, see Sec. 4. |
| 14 | pryzant2023 | OK | EMNLP 2023 (pages 7957-7968 from memory; UNVERIFIED, add only if checked). |
| 15 | guo2024 | OK | ICLR 2024. |
| 16 | yang2024b | OK | Verified Findings of EMNLP 2024, pp. 12163-12173, DOI 10.18653/v1/2024.findings-emnlp.709. |
| 17 | winata2023 | REPLACE | Verified arXiv 2306.10964 (2023-06-19, 9 pages), arXiv only; Semantic Scholar lists no peer-reviewed venue. It studies multilingual few-shot retrieval and does not establish general few-shot or reasoning-demonstration claims. Replace with Brown2020 (already [10]), Min2022 (already [18]), Wei2022 (already [8]), Wang2023 (already [21]) and Dong2024 (ICL survey, verified). Existing citations already suffice for the 3.3 claim; drop winata2023 everywhere. |
| 18 | min2022 | OK (qualify use) | Verified EMNLP 2022 pp. 11048-11064, DOI 10.18653/v1/2022.emnlp-main.759. Note: Min et al. show label correctness matters little and that format/label-space/input distribution matter. Use for "format and label space", not for "teach the reasoning process". |
| 19 | kojima2022 | OK | NeurIPS 2022. |
| 20 | wang2024 | OK | CoT without prompting, NeurIPS 2024 (vol. 37; bib says "38th conference", which is the NeurIPS 2024 edition: fine). |
| 21 | wang2023 | OK | Verified ACL 2023 Long, pp. 2717-2739, DOI 10.18653/v1/2023.acl-long.153. Directly supports "coherent, relevant reasoning steps matter" and demonstration-based CoT claims. |
| 22 | wang2024b | REPLACE | Verified Findings NAACL 2024, pp. 2243-2255, DOI 10.18653/v1/2024.findings-naacl.145. It is a domain-adaptation *training* method (role prompts used to build fine-tuning data while preserving general ability). It is not evidence that role assignment guides reasoning. Replace with Kong2024 (already cited), plus Zheng2024b (personas do not help) and Kim2024 for balance. |
| 23 | kim2024 | FIX | Verified arXiv 2408.08631 (2024-08, 14 pages); no peer-reviewed venue found. Add `journal = {arXiv preprint arXiv:2408.08631}`; authors match. It is the correct source for "poorly aligned personas can degrade reasoning". |
| 24 | zheng2024 | OK | Step-Back, ICLR 2024 (physics, chemistry, TimeQA, MuSiQue). Fits "knowledge-intensive reasoning". |
| 25 | codaforno2024 | FIX | Verified arXiv 2402.18225; ICML 2024 (PMLR 235) per bib, the arXiv page carries no venue note, so volume/pages UNVERIFIED. Abstract states "chain-of-thought prompting improves probabilistic reasoning" and "take-a-step-back prompting fosters model-based behaviors" (a behavioural-profile benchmark of 35 LLMs; model-based = planning/RL-style tasks). It does not show step-back improves knowledge-intensive reasoning. Keep, but with a precise statement (Sec. 4, edit E3). |
| 26 | yu2025 | FIX (use) | Verified Findings ACL 2025, pp. 6092-6111, DOI 10.18653/v1/2025.findings-acl.316. Concerns position bias in long-context settings (mitigation by scaling one hidden-state channel). Does not test short single-instruction prompts. |
| 27 | liu2024 | FIX (use) | Verified TACL vol. 12, pp. 157-173, DOI 10.1162/tacl_a_00638. Evidence is multi-document QA and key-value retrieval over long contexts (U-shaped curve). Not evidence about where to place instructions in short prompts. Add DOI. |
| 28 | wang2023b | OK | ICLR 2023. |
| 29 | ramnath2025 | FIX | Verified: published at EMNLP 2025 main, pp. 33066-33098, DOI 10.18653/v1/2025.emnlp-main.1681. Convert from arXiv-only. |
| 30 | cui2025 | FIX (use) | Verified Findings ACL 2025, pp. 22093-22111, DOI 10.18653/v1/2025.findings-acl.1140. A survey of heuristic *search* in instruction optimization; used in Sec. 2 as the source for "meta-prompting guides an LLM to generate/refine prompts using performance feedback". Acceptable as a survey pointer but weak for the specific definition; pair with Ye2024/Yang2024. |
| 31 | zhou2023 | OK | ICLR 2023. |
| 32 | yang2024 | OK | OPRO, ICLR 2024. |
| 33 | agarwal2025 | OK | Verified Findings ACL 2025, pp. 19974-20003, DOI 10.18653/v1/2025.findings-acl.1025. |
| 34 | khattab2024 | OK | ICLR 2024. |
| 35 | yuksekgonul2025 | OK | Verified Nature 639, pp. 609-616, DOI 10.1038/s41586-025-08661-4. Add DOI. |
| 36 | madaan2023 | OK | NeurIPS 2023. |
| 37 | shinn2023 | OK | NeurIPS 2023. |
| 38 | ye2024 | OK | Verified Findings ACL 2024, pp. 355-385, DOI 10.18653/v1/2024.findings-acl.21. Crossref spells the second author "Mohamed Ahmed" (arXiv: "Maxamed Axmed"); bib's "Ahmed, M." matches the proceedings. |
| 39 | chakma2026 | FIX | Verified arXiv 2606.29639 (submitted 2026-06-28), authors match. No venue found; add `journal = {arXiv preprint arXiv:2606.29639}`. |
| 40 | agrawal2026 | FIX | Verified ICLR 2026 (oral), arXiv 2507.19457; add eprint. Author list should be taken from arXiv (17 authors per search result: "Agrawal and 16 others"); bib lists 17, consistent. |
| 41 | lin2025 | FIX | Verified arXiv 2510.05921 (2025-10-07). arXiv author order is Lin, **Ruppik, van Niekerk, Shen, Heck, Lubis, Vukovic, Feng, Gasic**; bib puts Lubis second (wrong order). `@article` with no journal: change to `@misc` / arXiv preprint. No venue found. |
| 42 | nair2025 | FIX | Verified arXiv 2506.00178 (v1 2025-05-30, v2 2025-07-22), authors Nair, Banerjee, Mombaerts, Hagen, Borogovac (Amazon). The "KDD 2025 Workshop on Prompt Optimization" venue could NOT be verified: UNVERIFIED. Cite as arXiv with the eprint id unless the workshop listing is confirmed. |
| 43 | singh2026 | FIX | Verified arXiv 2602.00997 (2026-02-01); authors match; no venue found. |
| 44 | liu2025 | FIX | Verified arXiv 2502.04295 (2025-02-06); authors match; no venue found. Use is fine (content-format optimization). |
| 45 | li2025 | FIX | Verified arXiv 2502.11560 (2025-02-17); authors match; no venue found. |
| 46 | schulhoff2024 | FIX | Verified arXiv 2406.06608; the title changed between versions ("A Systematic Survey of Prompting Techniques" in v1 vs "Prompt Engineering Techniques" in the current abstract). Specify the version you read. Author list in bib is truncated mid-list with "others"; BibTeX will print et al. Fine. |
| 47 | sahoo2024 | FIX | Verified arXiv 2402.07927; authors match; no venue found. |
| 48 | boonstra2025 | FIX | Kaggle page `https://www.kaggle.com/whitepaper-prompt-engineering` fetched, but the fetch returned only the page title (no author/date/version). Author (Lee Boonstra, Google) confirmed by secondary sources only; the "February 2025" date is NOT confirmed: UNVERIFIED. Archive.org snapshot exists (20260905081837). |
| 49 | anthropic2025 | FIX | Canonical URL now `https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview` (301 from docs.anthropic.com), fetched 2026-10-20, no date shown. The page content has changed: it is now a short overview that defers to "Prompting best practices". Cite the best-practices page too if the role/XML/CoT guidance is what underpins the rubric. Snapshot 20261019210255 exists. |
| 50 | openai2025 | FIX | `https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api` returned HTTP 403 to the fetcher. Contents therefore NOT verified; no date seen. Do not state an access date for it. Archive.org snapshot exists (20260915104022). |
| 51 | microsoft2025 | FIX | Fetched 2026-10-20. Canonical URL `https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/prompt-engineering`, title "Prompt engineering techniques - Microsoft Foundry", `ms.date` 2026-05-13 (updated_at 2026-06-05). The page now says these techniques "aren't recommended for reasoning models like gpt-5 and o-series" and that on GPT-4/ChatGPT the "start with clear instructions" placement made no difference in their testing. Cite and disclose this (see edit E5). Snapshot 20251112071324 exists. |
| 52 | suzgun2023 | OK | Verified Findings ACL 2023, pp. 13003-13051, DOI 10.18653/v1/2023.findings-acl.824. |
| 53 | qwenteam2025 | FIX | arXiv 2505.09388 verified; authors on arXiv are "An Yang et al."; corporate author "Qwen Team" is accepted practice but give `author = {Yang, A. and others}` or keep "Qwen Team" with the arXiv id. Add `journal = {arXiv preprint arXiv:2505.09388}`. |

Other bib entries not in the printed list: `cobbe2021` OK (arXiv 2110.14168); `demar2006` (JMLR 7:1-30) OK from memory, UNVERIFIED this session; `dror2018`, `card2020` (verified arXiv 2010.06595 "To appear at EMNLP 2020"), `mizrahi2024` (verified TACL 12:933-949, DOI 10.1162/tacl_a_00681) OK. `openai2025b/c`, `googledeepmind2026`, `openai2026` are model/doc stubs without URLs; uncited, so delete or complete with URL if they are to be used. "Gemini 3.5 Flash-Lite" and the OpenAI documentation pages were not verified.

### Corrected / completed BibTeX for existing entries (paste over originals)

```bibtex
@article{goyal2023,
  author = {Goyal, T. and Li, J. J. and Durrett, G.},
  title = {News summarization and evaluation in the era of {GPT-3}},
  year = {2022},
  journal = {arXiv preprint arXiv:2209.12356},
}

@article{qin2025,
  author = {Qin, L. and Chen, Q. and Feng, X. and Wu, Y. and Zhang, Y. and Li, Y. and Li, M. and Che, W. and Yu, P. S.},
  title = {Large language models meet {NLP}: A survey},
  year = {2025},
  journal = {Frontiers of Computer Science},
  doi = {10.1007/s11704-025-50472-3},
  note = {arXiv:2405.12819},
}

@inproceedings{zhang2023,
  author = {Zhang, H. and Liu, X. and Zhang, J.},
  title = {Extractive summarization via {ChatGPT} for faithful summary generation},
  year = {2023},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages = {3270--3278},
  doi = {10.18653/v1/2023.findings-emnlp.214},
}

@article{zhang2025,
  author = {Zhang, X. and Tian, C. and Yang, X. and Chen, L. and Li, Z. and Petzold, L. R.},
  title = {{AlpaCare}: Instruction-tuned large language models for medical application},
  year = {2023},
  journal = {arXiv preprint arXiv:2310.14558},
}

@inproceedings{errica2025,
  author = {Errica, F. and Sanvito, D. and Siracusano, G. and Bifulco, R.},
  title = {What did {I} do wrong? {Q}uantifying {LLMs}' sensitivity and consistency to prompt engineering},
  year = {2025},
  booktitle = {Proc. 2025 Conf. Nations of the Americas Chapter of the Assoc. Comput. Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages = {1543--1558},
  doi = {10.18653/v1/2025.naacl-long.73},
}

@article{liu2023,
  author = {Liu, P. and Yuan, W. and Fu, J. and Jiang, Z. and Hayashi, H. and Neubig, G.},
  title = {Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing},
  year = {2023},
  journal = {ACM Computing Surveys},
  volume = {55},
  doi = {10.1145/3560815},
}

@inproceedings{reynolds2021,
  author = {Reynolds, L. and McDonell, K.},
  title = {Prompt programming for large language models: Beyond the few-shot paradigm},
  year = {2021},
  booktitle = {Extended Abstracts of the 2021 CHI Conference on Human Factors in Computing Systems},
  doi = {10.1145/3411763.3451760},
}

@inproceedings{zamfirescupereira2023,
  author = {Zamfirescu-Pereira, J. D. and Wong, R. Y. and Hartmann, B. and Yang, Q.},
  title = {Why {J}ohnny can't prompt: How non-{AI} experts try (and fail) to design {LLM} prompts},
  year = {2023},
  booktitle = {Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems},
  doi = {10.1145/3544548.3581388},
}

@inproceedings{mayilvaghanan2025,
  author = {Mayilvaghanan, K. and Nathan, V. and Kumar, A.},
  title = {{PROPEL}: Prompt optimization with expert priors for small and medium-sized {LLMs}},
  year = {2025},
  booktitle = {Proceedings of the 4th International Workshop on Knowledge-Augmented Methods for Natural Language Processing},
  pages = {272--302},
  doi = {10.18653/v1/2025.knowledgenlp-1.25},
}

@misc{kim2024,
  author = {Kim, J. and Yang, N. and Jung, K.},
  title = {Persona is a double-edged sword: Mitigating the negative impact of role-playing prompts in zero-shot reasoning tasks},
  year = {2024},
  eprint = {2408.08631},
  archivePrefix = {arXiv},
  note = {arXiv preprint},
}

@misc{lin2025,
  author = {Lin, H.-C. and Ruppik, B. M. and van Niekerk, C. and Shen, C.-H. and Heck, M. and Lubis, N. and Vukovic, R. and Feng, S. and Gasic, M.},
  title = {Prompt reinforcing for long-term planning of large language models},
  year = {2025},
  eprint = {2510.05921},
  archivePrefix = {arXiv},
}

@misc{ramnath2025,
  author = {Ramnath, K. and Zhou, K. and Guan, S. and Mishra, S. S. and Qi, X. and Shen, Z. and Wang, S. and Woo, S. and Jeoung, S. and Wang, Y. and Wang, H. and Ding, H. and Lu, Y. and Xu, Z. and Zhou, Y. and Srinivasan, B. and Yan, Q. and Chen, Y. and Ding, H. and Xu, P. and Cheong, L. L.},
  title = {A systematic survey of automatic prompt optimization techniques},
  year = {2025},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages = {33066--33098},
  doi = {10.18653/v1/2025.emnlp-main.1681},
}
% convert the ramnath2025 entry type to @inproceedings

@misc{nair2025,
  author = {Nair, A. and Banerjee, A. and Mombaerts, L. and Hagen, M. and Borogovac, T.},
  title = {Tournament of prompts: Evolving {LLM} instructions through structured debates and {Elo} ratings},
  year = {2025},
  eprint = {2506.00178},
  archivePrefix = {arXiv},
  note = {Workshop venue (KDD 2025) UNVERIFIED; cite as arXiv preprint unless confirmed},
}

@misc{agrawal2026,
  author = {Agrawal, L. A. and Tan, S. and Soylu, D. and Ziems, N. and Khare, R. and Opsahl-Ong, K. and Singhvi, A. and Shandilya, H. and Ryan, M. J. and Jiang, M. and Potts, C. and Sen, K. and Dimakis, A. G. and Stoica, I. and Klein, D. and Zaharia, M. and Khattab, O.},
  title = {{GEPA}: Reflective prompt evolution can outperform reinforcement learning},
  year = {2026},
  howpublished = {International Conference on Learning Representations (ICLR), Oral},
  eprint = {2507.19457},
  archivePrefix = {arXiv},
}

@misc{qwenteam2025,
  author = {Yang, A. and others},
  title = {Qwen3 technical report},
  year = {2025},
  eprint = {2505.09388},
  archivePrefix = {arXiv},
}

@misc{boonstra2025,
  author = {Boonstra, L.},
  title = {Prompt engineering},
  year = {2025},
  howpublished = {Google whitepaper, Kaggle},
  url = {https://www.kaggle.com/whitepaper-prompt-engineering},
  note = {Archived: https://web.archive.org/web/20260905081837/https://www.kaggle.com/whitepaper-prompt-engineering. Publication month UNVERIFIED.},
}

@misc{anthropic2025,
  author = {{Anthropic}},
  title = {Prompt engineering overview},
  year = {2025},
  howpublished = {Claude Developer Platform documentation},
  url = {https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview},
  note = {Accessed Oct. 20, 2026. Page revised since the version used for the rubric; archived: https://web.archive.org/web/20261019210255/https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview},
}

@misc{openai2025,
  author = {{OpenAI}},
  title = {Best practices for prompt engineering with the {OpenAI} {API}},
  year = {2025},
  howpublished = {OpenAI Help Center},
  url = {https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api},
  note = {Archived: https://web.archive.org/web/20260915104022/https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api. Live page returned HTTP 403 on Oct. 20, 2026; no access date asserted.},
}

@misc{microsoft2025,
  author = {{Microsoft}},
  title = {Prompt engineering techniques},
  year = {2026},
  howpublished = {Microsoft Foundry documentation, Microsoft Learn},
  url = {https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/prompt-engineering},
  note = {Page dated May 13, 2026. Accessed Oct. 20, 2026. Archived (earlier URL): https://web.archive.org/web/20251112071324/https://learn.microsoft.com/en-us/AZURE/AI-FOUNDRY/OPENAI/CONCEPTS/PROMPT-ENGINEERING},
}
```

(For `chakma2026`, `singh2026`, `liu2025`, `li2025`, `sahoo2024`, `schulhoff2024`, `colombo2024`: keep as is; they already carry correct eprint ids. `reynolds2021b`: delete.)

## 3. New / replacement BibTeX (all verified unless noted)

```bibtex
@inproceedings{vilar2023,
  author = {Vilar, D. and Freitag, M. and Cherry, C. and Luo, J. and Ratnakar, V. and Foster, G.},
  title = {Prompting {PaLM} for translation: Assessing strategies and performance},
  year = {2023},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {15406--15427},
  doi = {10.18653/v1/2023.acl-long.859},
}

@inproceedings{kamalloo2023,
  author = {Kamalloo, E. and Dziri, N. and Clarke, C. and Rafiei, D.},
  title = {Evaluating open-domain question answering in the era of large language models},
  year = {2023},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {5591--5606},
  doi = {10.18653/v1/2023.acl-long.307},
}

@inproceedings{dong2024,
  author = {Dong, Q. and others},
  title = {A survey on in-context learning},
  year = {2024},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages = {1107--1128},
  doi = {10.18653/v1/2024.emnlp-main.64},
}

@inproceedings{zheng2024b,
  author = {Zheng, M. and Pei, J. and Logeswaran, L. and Lee, M. and Jurgens, D.},
  title = {When ``a helpful assistant'' is not really helpful: Personas in system prompts do not improve performances of large language models},
  year = {2024},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2024},
  pages = {15126--15154},
  doi = {10.18653/v1/2024.findings-emnlp.888},
}

@inproceedings{zheng2023,
  author = {Zheng, L. and Chiang, W.-L. and Sheng, Y. and Zhuang, S. and Wu, Z. and Zhuang, Y. and Lin, Z. and Li, Z. and Li, D. and Xing, E. P. and Zhang, H. and Gonzalez, J. E. and Stoica, I.},
  title = {Judging {LLM}-as-a-judge with {MT-Bench} and {Chatbot Arena}},
  year = {2023},
  booktitle = {Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks Track)},
  doi = {10.52202/075280-2020},
}

@inproceedings{wang2024c,
  author = {Wang, P. and Li, L. and Chen, L. and Cai, Z. and Zhu, D. and Lin, B. and Cao, Y. and Kong, L. and Liu, Q. and Liu, T. and Sui, Z.},
  title = {Large language models are not fair evaluators},
  year = {2024},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {9440--9450},
  doi = {10.18653/v1/2024.acl-long.511},
}

@misc{panickssery2024,
  author = {Panickssery, A. and Bowman, S. R. and Feng, S.},
  title = {{LLM} evaluators recognize and favor their own generations},
  year = {2024},
  eprint = {2404.13076},
  archivePrefix = {arXiv},
  note = {arXiv preprint; no peer-reviewed venue found (NeurIPS 2024 listing UNVERIFIED)},
}

@inproceedings{huang2024,
  author = {Huang, J. and Chen, X. and Mishra, S. and Zheng, H. S. and Yu, A. W. and Song, X. and Zhou, D.},
  title = {Large language models cannot self-correct reasoning yet},
  year = {2024},
  booktitle = {The Twelfth International Conference on Learning Representations},
  eprint = {2310.01798},
  archivePrefix = {arXiv},
}

@inproceedings{zhang2024,
  author = {Zhang, Y. and Khalifa, M. and Logeswaran, L. and Kim, J. and Lee, M. and Lee, H. and Wang, L.},
  title = {Small language models need strong verifiers to self-correct reasoning},
  year = {2024},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  pages = {15637--15653},
  doi = {10.18653/v1/2024.findings-acl.924},
}

@inproceedings{sclar2024,
  author = {Sclar, M. and Choi, Y. and Tsvetkov, Y. and Suhr, A.},
  title = {Quantifying language models' sensitivity to spurious features in prompt design or: How {I} learned to start worrying about prompt formatting},
  year = {2024},
  booktitle = {The Twelfth International Conference on Learning Representations},
  eprint = {2310.11324},
  archivePrefix = {arXiv},
}

@inproceedings{wang2024d,
  author = {Wang, X. and Ma, B. and Hu, C. and Weber-Genzel, L. and R{\"o}ttger, P. and Kreuter, F. and Hovy, D. and Plank, B.},
  title = {``My answer is {C}'': First-token probabilities do not match text answers in instruction-tuned language models},
  year = {2024},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  pages = {7407--7416},
  doi = {10.18653/v1/2024.findings-acl.441},
}

@inproceedings{zheng2024c,
  author = {Zheng, C. and Zhou, H. and Meng, F. and Zhou, J. and Huang, M.},
  title = {Large language models are not robust multiple choice selectors},
  year = {2024},
  booktitle = {The Twelfth International Conference on Learning Representations},
  eprint = {2309.03882},
  archivePrefix = {arXiv},
}

@article{pineau2021,
  author = {Pineau, J. and Vincent-Lamarre, P. and Sinha, K. and Larivi{\`e}re, V. and Beygelzimer, A. and d'Alch{\'e}-Buc, F. and Fox, E. and Larochelle, H.},
  title = {Improving reproducibility in machine learning research (a report from the {NeurIPS} 2019 reproducibility program)},
  year = {2021},
  journal = {Journal of Machine Learning Research},
  volume = {22},
  number = {164},
  pages = {1--20},
}

@inproceedings{dodge2019,
  author = {Dodge, J. and Gururangan, S. and Card, D. and Schwartz, R. and Smith, N. A.},
  title = {Show your work: Improved reporting of experimental results},
  year = {2019},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing},
  pages = {2185--2194},
  doi = {10.18653/v1/D19-1224},
}

@inproceedings{bouthillier2021,
  author = {Bouthillier, X. and Delaunay, P. and Bronzi, M. and Trofimov, A. and Nichyporuk, B. and Szeto, J. and Mohammadi Sepahvand, N. and Raff, E. and Madan, K. and Voleti, V. and Ebrahimi Kahou, S. and Michalski, V. and Arbel, T. and Pal, C. and Varoquaux, G. and Vincent, P.},
  title = {Accounting for variance in machine learning benchmarks},
  year = {2021},
  booktitle = {Proceedings of Machine Learning and Systems 3 (MLSys 2021)},
}

@misc{badhe2026,
  author = {Badhe, S. and Shah, D.},
  title = {Prompt-level distillation: A non-parametric alternative to model fine-tuning for efficient reasoning},
  year = {2026},
  eprint = {2602.21103},
  archivePrefix = {arXiv},
  note = {Accepted at ACL 2026 Industry Track (per arXiv comment); proceedings pages UNVERIFIED},
}

@misc{shah2026,
  author = {Shah, D. and Badhe, S. and Kathrotia, N. and Tiwari, P.},
  title = {{CROP}: Token-efficient reasoning in large language models via regularized prompt optimization},
  year = {2026},
  eprint = {2604.14214},
  archivePrefix = {arXiv},
  note = {Accepted at ICLR 2026 Workshop on Logical Reasoning of Large Language Models (per arXiv comment)},
}

@misc{zhu2025,
  author = {Zhu, Z. and Zhou, H. and Feng, Z. and Li, T. and Deryl, C. J. J. and Onn, M. L. and Ng, G. W. and Mao, K.},
  title = {Rethinking prompt optimizers: From prompt merits to optimization},
  year = {2025},
  eprint = {2505.09930},
  archivePrefix = {arXiv},
  note = {No peer-reviewed venue found},
}

@misc{bsharat2023,
  author = {Bsharat, S. M. and Myrzakhan, A. and Shen, Z.},
  title = {Principled instructions are all you need for questioning {LLaMA-1/2}, {GPT-3.5/4}},
  year = {2023},
  eprint = {2312.16171},
  archivePrefix = {arXiv},
  note = {No peer-reviewed venue found},
}

@inproceedings{das2025,
  author = {Das, S. S. S. and Kamoi, R. and Pang, B. and Zhang, Y. and Xiong, C. and Zhang, R.},
  title = {{GReaTer}: Gradients over reasoning makes smaller language models strong prompt optimizers},
  year = {2025},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  eprint = {2412.09722},
  archivePrefix = {arXiv},
}

@misc{shim2026,
  author = {Shim, J. and Kim, D. W. and Kim, S. W. and others},
  title = {{POaaS}: Minimal-edit prompt optimization as a service to lift accuracy and cut hallucinations on on-device {sLLMs}},
  year = {2026},
  eprint = {2603.16045},
  archivePrefix = {arXiv},
  note = {Accepted at FEVER 2026 (per arXiv comment)},
}

@article{pandy2026,
  author = {P{\'a}ndy, {\'A}. and Lakatos, R. and Hajdu, A.},
  title = {Error-driven prompt optimization for arithmetic reasoning},
  year = {2026},
  journal = {IEEE Access},
  volume = {14},
  pages = {62570--62583},
  eprint = {2512.13323},
  archivePrefix = {arXiv},
}
```

Notes on items asked for:
- PE2 (`ye2024`) and PROPEL (`mayilvaghanan2025`) are already in the bib. Dror (`dror2018`) and Card (`card2020`) are already in the bib but uncited in the TeX; cite them.
- "Prompt-Level Distillation" is arXiv 2602.21103 by Badhe and Shah, ACL 2026 Industry Track per the arXiv comment; the reviewer's "Shah" co-author attribution is consistent. CROP is arXiv 2604.14214 (reviewer's id correct); authors are Shah, Badhe, Kathrotia, Tiwari; venue is an ICLR 2026 *workshop*, not the main conference.
- Dror 2018 and Card 2020 venues in bib were checked only against the arXiv record for Card (EMNLP 2020); Dror pages UNVERIFIED this session.
- I did not verify: GReaTer page numbers, Panickssery venue, Pándy/POaaS full author lists (arXiv first three only), Kocmi-style MT evaluation papers (not used).
- Not added (not verified or not relevant enough): Hu and Collier "Quantifying the persona effect in LLM simulations" (ACL 2024, verified, but about annotation simulation; optional).

## 4. Exact text edits for mismatched citations

All line numbers refer to `manuscript/latex_ieee/manuscript.tex`; the same sentences exist in `content_blocks.py` as tokens (translate to the reference_N tokens when applying; new keys need new reference_N entries and a regenerated `refkey_to_bibkey.json`).

**E1 (line 50, translation / summarization / QA).**
Original: `achieving strong performance on tasks such as translation \cite{goyal2023}, summarization \cite{qin2025}, and question answering \cite{zhang2023}`
Replace with: `achieving strong performance on tasks such as translation \cite{vilar2023}, summarization \cite{goyal2023}, and question answering \cite{kamalloo2023}` and cite the survey where the sentence continues, e.g. `...across NLP tasks \cite{qin2025}`. (`zhang2023` is then no longer cited and can be dropped; or retain it only for faithful extractive summarization.) Note Kamalloo et al. is an evaluation study of open-domain QA with LLMs, so phrase it as "evaluated on question answering", not "achieving strong performance" if you want to be conservative.

**E2 (line 52, role assignment in the intro).** `role assignment \cite{kong2024}` is acceptable. Add a counter-citation: `role assignment \cite{kong2024} (although its benefit is inconsistent \cite{zheng2024b,kim2024})`.

**E3 (line 82, Related Work, role prompting and step-back).**
Original: `Role prompting introduces contextual identities, ... to guide reasoning behavior and domain alignment \cite{wang2024b,kong2024}, although poorly aligned personas can degrade reasoning quality \cite{kim2024}. Step-back prompting further improves knowledge-intensive reasoning by encouraging the model to first abstract key principles before solving a task \cite{zheng2024,codaforno2024}.`
Replace with: `Role prompting introduces contextual identities, such as ``Act as a math teacher,'' and has been reported to improve zero-shot reasoning \cite{kong2024}; however, the evidence is mixed: poorly aligned personas can degrade reasoning \cite{kim2024}, and a large study across 2,410 factual questions and several open and proprietary models found that adding personas to system prompts does not reliably improve performance \cite{zheng2024b}. Step-back prompting improves knowledge-intensive reasoning by first abstracting key principles \cite{zheng2024}; a behavioural benchmark of 35 LLMs also found that take-a-step-back prompting fosters model-based behaviour in sequential decision tasks \cite{codaforno2024}.`
(Check the "2,410 questions" figure against Zheng et al. before keeping it; I did not verify that number, so delete the parenthetical count if unchecked. The safe wording is "a large-scale study found that adding personas does not reliably improve performance".) Drop `wang2024b`.

**E4 (line 80 and line 86, few-shot / demonstrations).**
Line 80: `few-shot prompting introduces demonstrations that improve generalization and output consistency \cite{winata2023}` -> `\cite{brown2020,dong2024}`.
Line 86: `...tend to reproduce that structure on novel inputs \cite{winata2023,min2022}` -> `\cite{wei2022,wang2023,min2022}`. Note Min2022 shows demonstrations help mostly through format, label space and input distribution, not label correctness; weaken "teach the reasoning process" accordingly or rely on `wei2022,wang2023` for that part.
Line 206: `...in few-shot and demonstration-based learning \cite{winata2023,min2022}` -> `\cite{brown2020,min2022,wang2023}`.

**E5 (lines 84 and 206, positional bias).**
Line 84 original: `because LLMs attend disproportionately to information placed near the beginning or end of a prompt \cite{yu2025,liu2024}, strategically ordering instructions and constraints can improve reasoning accuracy and instruction-following.`
Replace: `because LLMs attend disproportionately to information near the beginning or end of long contexts \cite{liu2024,yu2025}, and vendor guidance reports recency effects in prompts \cite{microsoft2025}, placing key instructions at the start or end is a common heuristic; whether it matters for the short prompts used here is an empirical question we test.`
Line 206 original: `The Conclusion criterion follows from evidence of positional bias in LLMs, whereby information near the start or end of a prompt receives more attention \cite{yu2025,liu2024}.`
Replace: `The Conclusion criterion follows from evidence of positional bias in long contexts \cite{liu2024,yu2025} and from industry guidance on repeating instructions at the end of a prompt \cite{microsoft2025}; we note that this evidence does not directly cover the short prompts studied here, and that Microsoft's own testing found no difference between instruction placements for GPT-4-class models.`
Line 86 also says closing restatement "exploits the same positional-attention effect": soften to "may exploit".

**E6 (lines 102, 206; role criterion).** `Role Prompting in contextual role assignment \cite{wang2024b,kong2024}` -> `\cite{kong2024}, with the caveat that gains are not consistent \cite{zheng2024b,kim2024}`. This also resolves the tension between the role criterion and [23] (Kim2024), which shows role prompts can hurt; the rubric should record that criterion 1 is the least-supported.

**E7 (line 52, "substantial trial-and-error \cite{mayilvaghanan2025}").** PROPEL is an optimizer paper. Better: `\cite{zamfirescupereira2023}` (non-experts struggle) already cited; change to `substantial trial-and-error \cite{zamfirescupereira2023,reynolds2021}` and keep PROPEL for the later "meta-prompting" claim.

**E8 (line 102, "Meta-prompting guides an LLM ... \cite{cui2025}").** Keep `cui2025` as a survey pointer but write "(see \cite{cui2025} for a survey)" and add `\cite{ye2024,yang2024b}` for the definition.

**E9 (line 194, industry sources).** Add a sentence: "Industry sources are cited as practitioner guidance, not as empirical evidence; they are living web documents, accessed 2026-10-20 where retrievable, and some now carry model-specific caveats (e.g. the Microsoft page states these techniques are not recommended for reasoning models)." The OpenAI page could not be fetched on 2026-10-20 (HTTP 403); cite the archive.org snapshot rather than asserting an access date.

**E10 (evaluation / LLM-judge use).** Where the paper describes the LLM-judged rubric scoring (Sec. 3 and the limitations), add: "LLM judges show position and self-preference biases \cite{zheng2023,wang2024c,panickssery2024}." Where the answer-extraction / delimiter-wrapped answers are discussed (line 86): add `\cite{sclar2024,wang2024d,zheng2024c}`. Where the paper discusses prompt-format sensitivity (lines 50, 54, 106): add `\cite{sclar2024}` next to `\cite{errica2025}`; and `\cite{mizrahi2024}` for multi-prompt evaluation. In the pre-registration / negative-result / power discussion (Sec. 5, revision protocol): `\cite{pineau2021,dodge2019,bouthillier2021,dror2018,card2020}`.
For the self-refinement limitation (null refinement effect, Sec. 5.1): `\cite{huang2024,zhang2024}` next to `\cite{madaan2023,shinn2023}`.

## 5. Related-work additions (<=200 words)

Prompt optimization with small or open-weight models is a growing line. PROPEL injects expert-derived principles as priors for small and medium-sized models \cite{mayilvaghanan2025}, GReaTer uses gradients over reasoning chains to let smaller models optimize prompts without a strong proprietary optimizer \cite{das2025}, and prompt-level distillation moves reasoning behaviour into prompts rather than weights \cite{badhe2026}, while regularized optimization targets token-efficient reasoning \cite{shah2026}. Meta-prompting formulations such as PE2 supply structured critique templates \cite{ye2024}, and recent analyses question what actually drives optimizer gains \cite{zhu2025}. Rubric-style refinement like ours should, however, be read against the limits of self-refinement: LLMs cannot reliably self-correct reasoning without external feedback \cite{huang2024}, and small models need strong verifiers to benefit from correction \cite{zhang2024}. Because our critic is an LLM judge, known position and self-preference biases \cite{zheng2023,wang2024c,panickssery2024} and sensitivity to prompt format and answer extraction \cite{sclar2024,wang2024d,zheng2024c} bound what a rubric score can certify. We therefore report fixed decision rules and variance-aware statistics \cite{dror2018,card2020,bouthillier2021,dodge2019,pineau2021}, so that a null result is interpretable rather than discarded.

(Every key above appears in section 3 or already exists in `cas-refs.bib`. Statements about each cited paper are limited to what its title/abstract metadata supports; I did not read full texts, so check claims such as GReaTer "without a strong proprietary optimizer" against the paper before keeping that phrase, UNVERIFIED.)

## Verification status summary

Verified this session (metadata from arXiv API / Crossref / official pages): all arXiv ids above, DOIs and pages for goyal-correction, zhang2023, wang2024b, kong2024, yu2025, liu2024, wang2023, min2022, ye2024, yang2024b, mayilvaghanan2025, cui2025, errica2025, agarwal2025, suzgun2023, ramnath2025, yuksekgonul2025, mizrahi2024, zamfirescupereira2023, reynolds2021, liu2023, vilar2023, kamalloo2023, dong2024, zheng2024b, zhang2024, wang2024c, wang2024d, dodge2019, pineau2021, bouthillier2021. Venue status of GEPA (ICLR 2026 oral), GReaTer (ICLR 2025), Huang (ICLR 2024), Sclar (ICLR 2024), Zheng MCQ (ICLR 2024 spotlight), MT-Bench (NeurIPS 2023 D&B) rests on arXiv/search metadata.
UNVERIFIED: Boonstra publication month; OpenAI page content (403); Nair KDD workshop venue; CogBench ICML volume/pages; Liu2023 article number; Pryzant EMNLP pages; Demsar/Dror page numbers; Panickssery venue; author lists truncated with "others" for Dong, Shim, Qwen.
