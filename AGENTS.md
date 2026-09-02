# AGENTS.md

MNIST handwritten-digit recognizer. Two unrelated halves live in this repo; neither has tests, a build system, or a requirements file.

## Layout & the big gotcha

- `main.py` — Python CNN (Conv + MaxPool + Dense) trained from scratch in pure NumPy. Prints/accuracy messages in Russian.
- `app.js` + `index.html` — browser drawing demo that fetches `weights.json` and runs inference in JS.
- MNIST `*.idx*-ubyte` files are committed and required — `main.py` has no download step.

**They are out of sync.** Committed `weights.json` and `app.js` implement the *old* two-layer fully-connected net: shapes `W1 (10,784)`, `b1 (10,1)`, `W2 (10,10)`, `b2 (10,1)`. `main.py` now trains a CNN whose params are conv filters `W1 (8,1,3,3)` + `b1 (8,1,1)` feeding a flattened `1352 → 10` dense head. `main.py` never writes weights and `app.js` has **no conv/maxpool JS code**, so regenerating weights from current `main.py` would silently break the demo. Don't assume the formats match.

## Running

- Use the local venv (no `requirements.txt` exists): `./venv/bin/python main.py`
  - Deps installed: numpy, pandas. Python 3.14.
- `main.py` runs the full MNIST load then trains on a small subset (1000 train / 200 test, 50 iters, lr 0.1) via hardcoded values in `__main__`.
- To open the demo, serve over HTTP (the `fetch('weights.json')` fails on `file://`): `python3 -m http.server` then visit `index.html`.

## Conventions & data shapes

- Weights/biases are serialized via `.tolist()` and stored as plain nested arrays. Biases are stored `(n,1)` arrays; the JS side reads them as `bias[i][0]` (`app.js:210`). Don't flatten biases.
- Network uses `(m, 1, 28, 28)` images (channel-first, `CHW`), unlike the flattened `784` the JS demo expects. ReLU + softmax.
- Python code is heavily commented and is written as a teaching script; keep that verbose style for `main.py`.
- No linter/formatter/typecheck config, no CI, no pre-commit hooks. To verify, actually run `main.py` (expect slow output).
