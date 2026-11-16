#!/usr/bin/env bash
# Domain-balanced training mixtures (v2): final (full) and held-out variants, prepared and tokenized.
set -e
export V45_DATA_ROOT=/data18/voyage_4_5_experiments
PY=/data/chandan.singh/voyage-4.5/.venv-aid/bin/python
cd "$(dirname "$0")/.."
$PY -I aidetect/01_prep.py --full --bal
$PY -I aidetect/02_tokenize.py --full --bal
$PY -I aidetect/01_prep.py --bal
$PY -I aidetect/02_tokenize.py --bal
