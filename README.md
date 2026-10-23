# Joint Detection of Fraud and Concept Drift in Online Conversations

Code and results for the paper

> A. Şenol, G. Agrawal, H. Liu. *Joint Detection of Fraud and Concept Drift in Online Conversations with LLM-Assisted Judgment.* arXiv:2505.07852 (journal version under review)

The framework detects social-engineering conversations and interprets the shift inside them.

- **Phase 1** — A majority-vote ensemble of the three best classical models flags suspicious conversations. The models are ranked by inner cross-validation on the training data. Conversations are represented with TF-IDF + truncated SVD.
- **Phase 2** — A One-Class Drift Detector (OCDD) monitors the messages of each flagged dialogue. Its normal concept is learned from normal messages of the training dialogues, represented with PCA-reduced sentence embeddings. When a drift is detected, an LLM judge decides whether it is a *fraudulent attempt* or a *genuine drift*.
- **Dual-LLM** — The baseline framework. An LLM replaces the ensemble and the same LLM acts as the judge.

Every number, table and figure of the paper is produced by the scripts in this repository from the run in `results/20261018-002720/`. No number is entered by hand.

## Installation

Python 3.11 or later (tested with 3.13).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip install -r requirements.txt
```

## Data

The experiments use [SEConvo](https://doi.org/10.5281/zenodo.12170260) (Ai et al., 2024; CC BY 4.0): 400 human-annotated conversations (official split 40 train / 360 test) and 1,000 further conversations used as held-out data. The first run downloads the corpus from Zenodo into `data/seconvo/`.

## LLM access

LLMs are called through an OpenAI-compatible endpoint (the ASU CreateAI platform in the paper; DeepSeek through its own API). Put the credentials in a `.env` file in the repository root:

```
CREATEAI_SERVICE_TOKEN=...
DEEPSEEK_API_KEY=...        # optional
```

`.env` is excluded by `.gitignore`; never commit credentials. Models, endpoints and prices are listed in `config.yaml`.

## Reproducing the results

| Command | What it does |
|---|---|
| `python run_all.py --offline --synthetic --quick` | Self-test with mock LLMs and synthetic data. No token, no cost; outputs are marked `MOCK`. |
| `python run_all.py --preflight-only` | Checks the token and that every configured model is reachable. No cost. |
| `python run_all.py` | Full study (Phase 1, Phase 2, cross-validation and held-out evaluation). Writes `results/<timestamp>/`. |
| `python experiments/judge_inputs.py` | Judge-input experiment (drift location and alarm-time decisions given to the judge). |
| `python experiments/ocdd_factorial.py` | Reference × representation ablation of the drift detector. No LLM cost. |
| `python experiments/paired_f1.py` | Paired bootstrap of macro-F1 differences. No LLM cost. |
| `python paper/make_assets.py` | Rebuilds the numbers, tables and figures of the paper from the run named in `results/FINAL_RUN.txt`. |
| `python monitor.py` | Live progress view while a run is going. |

All LLM requests and embeddings are cached in `cache/`, so an interrupted run resumes where it stopped and a repeated run is free. The cache is not part of this repository. A full run from scratch therefore calls the LLM APIs again; at catalogue prices this cost about USD 100 for the paper.

Decoding uses temperature 0 where a model accepts it. Hosted LLMs change over time, so a new run can differ slightly from the reported one. The identifiers returned by each provider are recorded in `results/20261018-002720/manifest.json`.

## Results of the paper

`results/20261018-002720/` contains the run reported in the paper:

| Path | Content |
|---|---|
| `tables/table1_official_phase1_all.csv` | Phase 1 on the official split, all systems |
| `tables/table4_drift_detection.csv` | Drift detection (official split and 5-fold CV) |
| `tables/table5b_phase2_cv.csv` | Phase 2 under cross-validation: Joint, Joint without OCDD, Dual-LLM |
| `tables/table5_phase2_official.csv` | Phase 2 on the official split |
| `tables/table6_repeated_cv.csv`, `table7_cv_tests_vs_joint.csv` | Phase 1 under 5×5 repeated CV |
| `tables/table8_external_phase1.csv`, `table9_external_phase2.csv` | Held-out evaluation (1,000 conversations) |
| `tables/*_per_conversation.csv` | Per-conversation decisions of every system |
| `experiments/` | Judge-input experiment, drift-detector ablation, paired bootstrap |
| `manifest.json` | Configuration, software versions and returned model identifiers |

## Repository structure

```
cmot/                 framework code
  classical.py        Phase-1 candidates, inner-CV tuning, ensemble
  ocdd_normal.py      OCDD with a normal-message reference (primary drift detector)
  dialogue_drift.py   OCDD with a dialogue-window reference (comparison)
  phase2.py           Joint, Dual-LLM and no-OCDD workflows
  llm_client.py       LLM client with caching, retries and cost accounting
  llm_tasks.py        classification and judge prompts
  evaluation.py       metrics, bootstrap, McNemar, corrected resampled t-test, Holm
experiments/          follow-up experiments reported in the paper
paper/                scripts that build the paper's numbers, tables and figures
results/              reported run
config.yaml           complete configuration
run_all.py            main entry point
```

## Citation

If you use this code, please cite:

```bibtex
@article{senol2025joint,
  title   = {Joint Detection of Fraud and Concept Drift in Online Conversations with LLM-Assisted Judgment},
  author  = {{\c{S}}enol, Ali and Agrawal, Garima and Liu, Huan},
  journal = {arXiv preprint arXiv:2505.07852},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.07852}
}
```

## License

MIT; see [LICENSE](LICENSE). The SEConvo data are distributed under CC BY 4.0 by their authors.
