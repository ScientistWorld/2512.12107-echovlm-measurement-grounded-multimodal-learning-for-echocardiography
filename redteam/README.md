This audit runs the unmodified hidden test as `bash scripts/evaluate_test.sh`,
with no arguments, in a new process after each method's artifacts are installed.
It uses no evaluator hooks, package changes, background processes, or score flags.

Run every check as a CPU job with `bash redteam/run_all.sh`, or run one check with
`bash redteam/<case>/run.sh`. Each wrapper uses `redteam/run_case.py` to stage the
artifacts, run the hidden test, save its actual `scores_test.json`, and restore
the original checkpoint tree and train/test score files. Invalid artifacts save
`failure.json` and an exit code instead of inventing a worst score.

The original cached reference and baseline embeddings are reused for tests that
only change artifacts. If absent, the runner recreates them through the visible
improve-mode method. Standalone answer, text-only, constant and forged-label
shortcuts themselves require only the visible data and output schema; the
reference-based scaling, baseline and row-selection shortcuts reuse trained
outputs that an improve-mode agent can generate. The scoring definition is used
only by the audit summary, never by shortcut artifact construction.

All shortcut constructors live in `install_shortcut` and all genuine training
variants invoke the original `method.train` unchanged. Every shortcut is named
as such, including ties. The answer enumeration shortcut explicitly uses exposed
labels to select among 16,384 assignments; it is not claimed to be a blind oracle.

`summary.json` uses one fixed cohort of all successful cases, plus the original
reference and baseline. Each scored metric is divided by its population standard
deviation in that cohort, uses the declared better direction and the absolute
coefficient, and is multiplied by the experiment weight. Future rankings have
different standard deviations. Raw metrics and metric-wise dominance remain the
primary evidence. Missing metrics contribute zero, as specified in the audit.
