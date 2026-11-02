# Reference audit, round 2 (2026-11-02)

Scope: every entry of `latex_ieee/cas-refs.bib` with year 2025/2026 or arXiv/web/whitepaper status, plus the three sentences the reviewer questioned. Sources: arXiv API, ACL Anthology (page + .bib export), PMLR, Crossref, NeurIPS proceedings, Amazon Science, web search. Round-1 findings (`reference_audit.md`) are not repeated unless I found an error in them. Nothing was edited; `cas-refs.bib` is unchanged.

Status legend: OK = confirmed, no change; CORRECT = metadata fix; UPDATE-TO-PEER-REVIEWED = a peer-reviewed version exists; UNVERIFIED = could not confirm.

## 0. Errors found in round 1 / in the current bib

1. `ramnath2025`: the bib has `@misc` with `booktitle`. BibTeX/IEEEtran drops `booktitle` in `@misc`, so the printed reference has no venue. It must be `@inproceedings`. Pages: the ACL Anthology record and its .bib export give 33078--33110, while Crossref gives 33066--33098 (and round 1 copied the Crossref value). I use the Anthology value because it is the publisher of record. Crossref and Anthology disagree, so re-check against the PDF if you want certainty.
2. `kim2024`: the paper is now peer reviewed (Findings of IJCNLP-AACL 2025) under a changed title. Round 1 said "no peer-reviewed venue found"; that is now outdated.
3. `zhu2025`: now EACL 2026 (round 1 said no venue found). The bib's author list is also wrong: it has 8 authors with "Mao, K." as 8th but names "Onn, M. L." The correct 6th/7th/8th authors are Lee Onn Mak, Gee Wah Ng, Kezhi Mao.
4. `wang2024`: the bib says "38th International Conference on NeurIPS" with no volume or pages. The proceedings are "Advances in Neural Information Processing Systems 37" (NeurIPS 2024 is the 38th conference, the volume is 37).
5. `codaforno2024`: arXiv v1 says 35 LLMs; the ICML version (PMLR 235) says 40 LLMs. The manuscript sentence says "35 LLMs". Change it to 40, or drop the number.

## (a) Table

Printed numbers in the submitted PDF cannot be recomputed exactly here. The reviewer's [43], [45], [47], [48], [51]-[54] fall in the round-1 mapping region singh2026, li2025, sahoo2024, boonstra2025, microsoft2025, suzgun2023(not 2026), qwenteam2025. I checked all of them regardless of number.

| bib key | status | corrected fields / finding |
|---|---|---|
| singh2026 | OK (arXiv only) | arXiv 2602.00997v1, 2026-02-01, Singh, Yadav, Blanco, title exact. No venue found (one search; a listing hints at OpenReview but no decision). Keep as arXiv preprint. |
| li2025 | OK (arXiv only) | arXiv 2502.11560v1, 2025-02-17; Li, Wang, Li, Jin; title exact; no venue found. |
| sahoo2024 | OK (arXiv only) | arXiv 2402.07927 (v1 2024-02-05, v2 2025-03-16); six authors match; no venue found. Optionally add "(v2, Mar. 2025)". |
| schulhoff2024 | CORRECT | arXiv 2406.06608 v6 (2025-02-26) title is "The Prompt Report: A Systematic Survey of Prompt Engineering Techniques" (bib has "...of prompting techniques"). 31 authors; bib lists 19 plus "others" (fine). No venue found. Year 2024 = first submission, acceptable; keep and add `note = {arXiv v6, Feb. 2025}`. |
| boonstra2025 | UNVERIFIED (date) | Author Lee Boonstra (Google) and Kaggle URL confirmed by secondary sources. Press coverage dates release to April 2025; the "February 2025" in round 1 and the cover date could not be confirmed. Keep `year = {2025}`, no month. |
| anthropic2025, openai2025, microsoft2025 | OK (web pages) | Not re-fetched; round-1 URLs, archive links and caveats stand. OpenAI page content remains unverified (HTTP 403 in round 1). |
| qwenteam2025 | OK | arXiv 2505.09388v1, 2025-05-14, 60 authors, first An Yang; no peer-reviewed version (DBLP lists CoRR only). Keep. |
| chakma2026 | OK (arXiv only) | arXiv 2606.29639v1, 2026-06-28; authors Chakma, Surdeanu, Blanco and title match the arXiv API; no venue found. |
| agrawal2026 | CORRECT (type) | arXiv 2507.19457v2: 17 authors, "Accepted to ICLR 2026 (Oral)". Change `@misc/howpublished` to `@inproceedings` with proper booktitle (ICLR 2026 = Fourteenth ICLR). Proceedings/OpenReview URL not checked. |
| lin2025 | OK (arXiv only) | arXiv 2510.05921 (v3), 9 authors, order Lin, Ruppik, van Niekerk, Shen, Heck, Lubis, ... matches bib; no venue found. |
| nair2025 | UPDATE-TO-PEER-REVIEWED (workshop) | Venue now confirmed on the authors' Amazon Science page: KDD 2025 Workshop on Prompt Optimization (a workshop, not a main-track proceeding). arXiv 2506.00178v2, 5 authors match. Add booktitle. |
| liu2025 | OK (arXiv only) | arXiv 2502.04295v3, 9 authors, title exact; no venue found. |
| zhu2025 | UPDATE-TO-PEER-REVIEWED | EACL 2026 (19th EACL, Vol. 1 Long), Rabat, March 2026, pp. 864--892, DOI 10.18653/v1/2026.eacl-long.38, verified on the ACL Anthology page. |
| kim2024 | UPDATE-TO-PEER-REVIEWED | Findings of IJCNLP-AACL 2025, Mumbai, Dec. 2025, pp. 848--862, DOI 10.18653/v1/2025.findings-ijcnlp.51, new title. Verified on ACL Anthology. |
| ramnath2025 | CORRECT | See section 0 item 1: `@inproceedings`, pp. 33078--33110, 21 authors (bib list matches). |
| badhe2026 | OK, add DOI | ACL 2026 Industry Track (Vol. 6), San Diego, July 2026, pp. 2131--2147, DOI 10.18653/v1/2026.acl-industry.142. Authors Badhe, Shah. Bib already correct except for DOI/volume/address. |
| shah2026 (CROP) | OK | arXiv 2604.14214, 4 authors match; "Accepted at ICLR 2026 Workshop on Logical Reasoning of LLMs" (arXiv comment). Optional note. |
| shim2026 | OK (UNVERIFIED proceedings) | arXiv 2603.16045, 7 authors, first three match; "Accepted at FEVER 2026" per arXiv comment; proceedings entry not found. |
| pandy2026 | OK | arXiv 2512.13323 journal_ref: IEEE Access vol. 14, pp. 62570--62583, 2026; matches bib. |
| das2025 | OK | arXiv 2412.09722 "ICLR 2025 Camera Ready"; 6 authors match. |
| panickssery2024 | UPDATE-TO-PEER-REVIEWED | NeurIPS 2024 main track (Oral), Advances in NeurIPS 37, DOI 10.52202/079017-2197. Page numbers not obtained. |
| bsharat2023 | OK (arXiv only) | arXiv 2312.16171 (v2 2024-01-18), 3 authors; one search, no venue found. |
| colombo2024 | OK (arXiv only) | arXiv 2403.03883, 11 authors; no venue found. |
| zhang2025 (AlpaCare) | CORRECT | arXiv 2310.14558 (v6): title in arXiv API "AlpaCare: Instruction-tuned Large Language Models for Medical Application"; a later version carries "Instruction Fine-tuned LLMs for Medical Applications" and states ICLR 2025 SCI-FM workshop (from search snippet, UNVERIFIED). Keep `year = {2023}` + arXiv id; no archival venue. |
| goyal2023 | OK | arXiv 2209.12356v2, 2022-09-26, 3 authors; no venue. |
| winata2023 | OK (uncited per round 1) | arXiv 2306.10964, 4 authors match. Delete if uncited. |
| cobbe2021 | OK | arXiv 2110.14168, 12 authors. |
| wang2024 | CORRECT | NeurIPS 2024 = Advances in NeurIPS 37, pp. 66383--66409, DOI 10.52202/079017-2123 (Crossref). arXiv 2402.10200 (Wang, Zhou). |
| wang2023 | OK, add DOI | ACL 2023 Long, pp. 2717--2739, DOI 10.18653/v1/2023.acl-long.153 (ACL Anthology). Authors match. |
| codaforno2024 | CORRECT | ICML 2024, PMLR 235:9076--9108 (PMLR page). arXiv 2402.18225 v1 abstract: 35 LLMs; PMLR abstract: 40 LLMs. |
| openai2025b/c, googledeepmind2026, openai2026, luna2026 | UNVERIFIED | Model/doc stubs; not re-checked (Gemini 3.5 Flash-Lite, GPT-6 Luna and OpenAI deprecation pages not accessed). |
| pryzant2023 | OK, add pages | ACL Anthology: EMNLP 2023 main, pp. 7957--7968, DOI 10.18653/v1/2023.emnlp-main.494 (resolves a round-1 UNVERIFIED). |
| liu2023 | OK | Crossref: ACM Comput. Surv. 55(9), pp. 1--35, no article number, DOI 10.1145/3560815. Bib's "Article 195" should not be added; add `number = {9}` and optional `pages = {1--35}`. |

## (b) Answers to step 2

Printed numbers: reviewer's [25] = wang2024, [26] = wang2023, [30] = codaforno2024 (consistent with the sentences below; round-1 numbering was offset).

### wang2024 and wang2023 (manuscript.tex, line 78)

Sentence: "Chain-of-thought prompting decomposes multistep problems \cite{wei2022,kojima2022}, and coherent, relevant reasoning steps matter more than length \cite{wang2024,wang2023}."

- wang2023 (Wang et al., ACL 2023) supports it, with a qualification. Its abstract: CoT retains about 80-90% of performance with invalid demonstrated steps, while relevance of the rationale to the query and correct step order matter more. It supports "relevant, coherent, correctly ordered steps matter more than step validity". The abstract I read does not mention length; "more than length" is not established.
- wang2024 (Wang and Zhou, NeurIPS 2024) does not support it. Its abstract is about eliciting CoT paths from decoding (top-k alternative tokens) without prompts, plus a confidence signal. It says nothing about coherence, relevance or length. The reviewer is right: cite wang2023 only.
- Proposed sentence: "Chain-of-thought prompting decomposes multistep problems \cite{wei2022,kojima2022}, and the relevance of the reasoning steps to the question and their correct order matter more than the validity of each step \cite{wang2023}."
  (If you want "length" in the claim, I did not verify a source; Jin et al. 2024, "The impact of reasoning step length on LLMs", is a candidate but is UNVERIFIED here and I did not check it.)
- If wang2024 should stay in the paper, a supportable use is "CoT reasoning can be elicited from pretrained models by changing decoding alone, without prompts \cite{wang2024}", which the abstract does support; otherwise remove the entry from the bib.

### codaforno2024 (line 184)

Sentence: "Step-back prompting abstracts key principles before solving \cite{zheng2024}; a behavioural benchmark of 35 LLMs found that it fosters model-based behaviour in sequential decision tasks \cite{codaforno2024}, although this is weaker evidence for knowledge-intensive reasoning than the original step-back results."

- CogBench (ICML 2024) is a behavioural benchmark of ten metrics from seven cognitive-psychology experiments. The PMLR abstract says 40 LLMs (arXiv v1: 35), and that "take-a-step-back prompting promoted model-based behavior". So the clause is accurate, apart from the count, and the sentence already hedges. But it is only tangential support for a general step-back claim, and the reviewer's complaint is fair.
- Recommendation: delete the codaforno2024 clause and cite only zheng2024 (ICLR 2024), then drop the bib entry, unless the paper needs a second step-back source. If you keep it, use:
  "Step-back prompting abstracts key principles before solving \cite{zheng2024}; in a behavioural benchmark of 40 LLMs, take-a-step-back prompting was also associated with more model-based behaviour in sequential decision tasks \cite{codaforno2024}, which is indirect evidence at best for knowledge-intensive reasoning."
- I did not find (and so do not propose) a verified second step-back reference.

## (c) Corrected BibTeX

```bibtex
@inproceedings{wang2024,
  author = {Wang, X. and Zhou, D.},
  title = {Chain-of-thought reasoning without prompting},
  year = {2024},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  pages = {66383--66409},
  doi = {10.52202/079017-2123},
}

@inproceedings{wang2023,
  author = {Wang, B. and Min, S. and Deng, X. and Shen, J. and Wu, Y. and Zettlemoyer, L. and Sun, H.},
  title = {Towards understanding chain-of-thought prompting: An empirical study of what matters},
  year = {2023},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {2717--2739},
  doi = {10.18653/v1/2023.acl-long.153},
}

@inproceedings{codaforno2024,
  author = {Coda-Forno, J. and Binz, M. and Wang, J. X. and Schulz, E.},
  title = {{CogBench}: A large language model walks into a psychology lab},
  year = {2024},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {235},
  pages = {9076--9108},
}

@inproceedings{kim2024,
  author = {Kim, J. and Yang, N. and Jung, K.},
  title = {Persona is a double-edged sword: Rethinking the impact of role-play prompts in zero-shot reasoning tasks},
  year = {2025},
  booktitle = {Findings of the Association for Computational Linguistics: IJCNLP-AACL 2025},
  pages = {848--862},
  doi = {10.18653/v1/2025.findings-ijcnlp.51},
}

@inproceedings{ramnath2025,
  author = {Ramnath, K. and Zhou, K. and Guan, S. and Mishra, S. S. and Qi, X. and Shen, Z. and Wang, S. and Woo, S. and Jeoung, S. and Wang, Y. and Wang, H. and Ding, H. and Lu, Y. and Xu, Z. and Zhou, Y. and Srinivasan, B. and Yan, Q. and Chen, Y. and Ding, H. and Xu, P. and Cheong, L. L.},
  title = {A systematic survey of automatic prompt optimization techniques},
  year = {2025},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages = {33078--33110},
  doi = {10.18653/v1/2025.emnlp-main.1681},
}

@inproceedings{zhu2025,
  author = {Zhu, Z. and Zhou, H. and Feng, Z. and Li, T. and Deryl, C. J. J. and Mak, L. O. and Ng, G. W. and Mao, K.},
  title = {Rethinking prompt optimizers: From prompt merits to optimization},
  year = {2026},
  booktitle = {Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {864--892},
  doi = {10.18653/v1/2026.eacl-long.38},
}

@inproceedings{badhe2026,
  author = {Badhe, S. and Shah, D.},
  title = {Prompt-level distillation: A non-parametric alternative to model fine-tuning for efficient reasoning},
  year = {2026},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track)},
  pages = {2131--2147},
  doi = {10.18653/v1/2026.acl-industry.142},
}

@inproceedings{nair2025,
  author = {Nair, A. and Banerjee, A. and Mombaerts, L. and Hagen, M. and Borogovac, T.},
  title = {Tournament of prompts: Evolving {LLM} instructions through structured debates and {Elo} ratings},
  year = {2025},
  booktitle = {KDD 2025 Workshop on Prompt Optimization},
  note = {arXiv:2506.00178},
}

@inproceedings{agrawal2026,
  author = {Agrawal, L. A. and Tan, S. and Soylu, D. and Ziems, N. and Khare, R. and Opsahl-Ong, K. and Singhvi, A. and Shandilya, H. and Ryan, M. J. and Jiang, M. and Potts, C. and Sen, K. and Dimakis, A. G. and Stoica, I. and Klein, D. and Zaharia, M. and Khattab, O.},
  title = {{GEPA}: Reflective prompt evolution can outperform reinforcement learning},
  year = {2026},
  booktitle = {The Fourteenth International Conference on Learning Representations (ICLR 2026), Oral},
  note = {arXiv:2507.19457},
}

@inproceedings{panickssery2024,
  author = {Panickssery, A. and Bowman, S. R. and Feng, S.},
  title = {{LLM} evaluators recognize and favor their own generations},
  year = {2024},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  doi = {10.52202/079017-2197},
}

@misc{schulhoff2024,
  author = {Schulhoff, S. and Ilie, M. and Balepur, N. and Kahadze, K. and Liu, A. and Si, C. and Li, Y. and Gupta, A. and Han, H. and Schulhoff, S. and Dulepet, P. S. and Vidyadhara, S. and Ki, D. and Agrawal, S. and Pham, C. and Kroiz, G. C. and Li, F. and Tao, H. and Resnik, P. and others},
  title = {The prompt report: A systematic survey of prompt engineering techniques},
  year = {2024},
  eprint = {2406.06608},
  archivePrefix = {arXiv},
  howpublished = {arXiv preprint arXiv:2406.06608},
  note = {v6, Feb. 2025},
}

@inproceedings{pryzant2023,
  author = {Pryzant, R. and Iter, D. and Li, J. and Lee, Y. T. and Zhu, C. and Zeng, M.},
  title = {Automatic prompt optimization with ``gradient descent'' and beam search},
  year = {2023},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages = {7957--7968},
  doi = {10.18653/v1/2023.emnlp-main.494},
}

@article{liu2023,
  author = {Liu, P. and Yuan, W. and Fu, J. and Jiang, Z. and Hayashi, H. and Neubig, G.},
  title = {Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing},
  year = {2023},
  journal = {ACM Computing Surveys},
  volume = {55},
  number = {9},
  pages = {1--35},
  doi = {10.1145/3560815},
}

@article{zhang2025,
  author = {Zhang, X. and Tian, C. and Yang, X. and Chen, L. and Li, Z. and Petzold, L. R.},
  title = {{AlpaCare}: Instruction-tuned large language models for medical application},
  year = {2023},
  journal = {arXiv preprint arXiv:2310.14558},
}
```

Notes: the `kim2024` key and `zhu2025`/`nair2025` keys are kept so that `\cite{}` calls need no change. IEEEtran prints `year` as given, so `zhu2025` with `year = {2026}` and `kim2024` with `year = {2025}` are intentional; rename keys if you prefer consistency. For the `@inproceedings` entries that carry the proceedings volume only through `doi`, IEEEtran will print the DOI without a volume; this is acceptable.

## Items still UNVERIFIED after this round

Boonstra publication month; OpenAI help page content; Anthropic/Microsoft pages not re-fetched; Panickssery page range; AlpaCare workshop acceptance (search snippet only); shim2026 proceedings; Gemini/GPT-6 Luna/OpenAI deprecation stubs; any peer-reviewed version of singh2026, li2025, liu2025, sahoo2024, lin2025, chakma2026, bsharat2023, colombo2024, winata2023 beyond the searches run (one search each at most; none found). Dror/Card/Demsar page numbers (uncited entries) not re-checked.
