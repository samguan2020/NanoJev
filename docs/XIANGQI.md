# Human-vs-NanoJev Xiangqi (Chinese chess)

A small human-vs-machine Xiangqi app built on top of this repo's model-serving
pattern: a pure-Python rules engine owns the board, and NanoJev is asked a
single `choice` question per AI turn — "pick the best move among these legal
candidates." No chain-of-thought, no move generation by the model; the model
only scores a fixed candidate list produced by the rules engine.

## Start it (two servers)

```bash
# Terminal 1: the model service (loads Qwen3-0.6B + a checkpoint onto the GPU)
python scripts/serve_decisions.py --checkpoint-dir runs/xiangqi_full_v3 --web-root web --port 8765

# Terminal 2: the game server (owns board state, calls terminal 1 for AI moves)
python scripts/serve_xiangqi.py --web-root web --port 8766 --nanojev-url http://127.0.0.1:8765
```

Then open **http://127.0.0.1:8766/xiangqi.html**.

`--checkpoint-dir` can point at any of:
- `checkpoints/NanoJev` — the original maze/toy-support checkpoint, never trained on Xiangqi.
- `runs/xiangqi_full_v1` — first fine-tune: 1200 Pikafish-labeled positions, 150 steps (frozen backbone). Dev target CE 2.374.
- `runs/xiangqi_full_v2` — same 1200 positions, 430 steps. Dev target CE 2.355, test 1.865, ood 1.997. Best step was 370/430 — later steps drifted back up, an early overfitting signal given the fixed 1200-example pool.
- `runs/xiangqi_full_v3` — **current default above**: 6000 fresh Pikafish-labeled positions (data/xiangqi_v3), 2040 steps, batch-questions=4. Dev target CE **2.053** (best of the three so far), calibration 2.320, test 2.270, ood 2.237 — note these last three aren't directly comparable to v1/v2's numbers since they're a different, freshly-generated eval set, not the same held-out questions. Best step was 1460/2040. Peak true GPU allocation stayed at 2.77GB, identical to v1/v2, even with 5x the data — the higher `nvidia-smi` readings seen mid-run were the CUDA caching allocator holding reserved-but-unused memory, not a real increase in need. See "Training a real checkpoint" below for how these were made and what would likely help most next (more/fresher labeled positions, not more steps on the same pool).

`serve_xiangqi.py` loads no model weights itself — it just calls the already-running
`serve_decisions.py` over HTTP, so the two servers share one GPU-resident model instead
of loading it twice. If `serve_decisions.py` isn't reachable, `serve_xiangqi.py` still
runs and falls back to picking a uniformly random legal move (clearly labeled
`"source": "random_fallback"` in the response), so the board/rules/UI can be exercised
without a GPU.

## How to play

1. Pick which side you play (red moves first) and click **开始新对局**.
2. Click one of your own pieces — legal destinations light up green.
3. Click a highlighted square to move there.
4. If it's the AI's turn, the status bar shows "AI 思考中" while it thinks
   (typically ~8-10 seconds on a 4GB GPU); the move log shows what it played,
   its source (`nanojev` vs `random_fallback`), and the model's probability
   for that choice.
5. The game ends on checkmate or stalemate (both are a loss for the side with
   no legal moves, per Xiangqi rules — stalemate is not a draw here).

## Tech stack

| Layer | What | Notes |
|---|---|---|
| Model | Qwen3-0.6B backbone + NanoJev decision head | `checkpoints/NanoJev` (HF `C-Tianyu/NanoJev`); untrained for Xiangqi specifically — see caveat below |
| Inference | PyTorch 2.6 (`cu124`), Transformers, safetensors | `scripts/serve_decisions.py`, `scripts/predict_toy_decisions.py`; CUDA required (no CPU path in this predictor) |
| Rules engine | Pure Python, stdlib only | `scripts/xiangqi_board.py` — board dict, move generation, check/mate detection; unit-tested with `unittest` |
| NanoJev bridge | Pure Python, stdlib only | `scripts/xiangqi_notation.py` — board/move <-> text, builds the `choice` question payload |
| Game server | Python `http.server` (no framework) | `scripts/serve_xiangqi.py` — holds one in-memory game, validates human moves, calls the model service for AI moves |
| Frontend | Vanilla HTML/CSS/JS, no build step, no framework | `web/xiangqi.html` — `fetch()` against the game server's JSON API, absolutely-positioned `div`s for the board grid |
| Transport | JSON over HTTP, localhost only | Two independent processes talk over `127.0.0.1`; the browser talks only to the game server |

Run the rules/notation tests any time with:
```bash
python -m unittest discover -s scripts -p test_xiangqi_board.py -v
python -m unittest discover -s scripts -p test_xiangqi_notation.py -v
```

## Workflow: what happens on one AI turn

1. Human clicks a destination square → browser `POST /api/move {from, to}` to `serve_xiangqi.py`.
2. Server validates the move against `xiangqi_board.legal_moves(...)`, applies it, flips `side_to_move`.
3. Server calls `xiangqi_board.legal_moves(...)` again for the AI's side. If there are
   more than `MAX_CANDIDATES` (12), it prunes: keep all capturing moves, fill the rest
   with a random sample of quiet moves. (Reason below.)
4. Server builds one NanoJev request: `state` = the whole board rendered as text,
   one `choice` question whose `criteria` are the pruned candidate moves.
5. Server `POST`s that to `serve_decisions.py`'s `/api/evaluate` (the model already
   loaded in that other process), gets back a probability per candidate.
6. Server applies the highest-probability move, appends it to the move log, checks
   `game_status(...)` for checkmate/stalemate.
7. Server returns the full new state (+ what the AI played, its source, its probability)
   to the browser in the same HTTP response that started at step 1 — the browser just re-renders.

### Why the pruning in step 3 exists

This repo's reference inference implementation re-encodes the *entire* state text for
every candidate path (no shared-prefix batching). The opening position has 44 legal
moves and a ~190-token board description; sending all of them as one `choice` question
made a single AI turn take minutes on a 4GB GPU instead of single-digit seconds. Capping
the candidate list (captures first, then a random sample of quiet moves) is the same
"code narrows the options, the model judges what's left" split this repo's maze/snake
demos already use — not a workaround bolted on after the fact.

## Training a real checkpoint (Pikafish-labeled data + frozen-backbone fine-tune)

The base `checkpoints/NanoJev` checkpoint (maze/toy-support trained) has no real Xiangqi
judgment — it answers the `choice` question fluently but plays weak, sometimes nonsensical
moves. `runs/xiangqi_full_v1`/`v2` are fine-tunes built from this repo that measurably
(if modestly) improve on that. The pipeline:

1. **Engine**: [official-pikafish/Pikafish](https://github.com/official-pikafish/Pikafish)
   (GPL, free, UCI protocol), extracted to `engines/`. `scripts/pikafish_engine.py` is a
   minimal UCI client — note UCI xiangqi squares (`<file a-i><rank 0-9>`) are exactly this
   repo's `(row, col)`, so no coordinate translation beyond letter<->index is needed.
2. **Data generation** — `scripts/build_xiangqi_data.py`: self-play games pick a uniformly
   random legal move each ply (seeded); at sampled positions, Pikafish's best move becomes
   `gold` and the (pruned, same `prune_moves` as the live server) legal move list becomes
   the `choice` criteria. Positions with only one legal move are skipped (a `choice` needs
   >=2 options). Example:
   ```bash
   python scripts/build_xiangqi_data.py --output-dir data/xiangqi_v2 \
     --train-count 1200 --dev-count 10 --calibration-count 10 --test-count 15 --ood-count 15
   ```
3. **Validate** before spending GPU time: `python scripts/train_pipeline_decisions.py --input data/xiangqi_v2 --validate-only`
4. **Train with `--freeze-backbone`**: full-parameter fine-tuning of the 0.6B backbone
   needs ~9.6GB of AdamW state (params+grad+2 moments) and OOMs on a 4GB GPU — confirmed
   experimentally. `--freeze-backbone` (added to `train_pipeline_decisions.py` for this)
   keeps the backbone out of the optimizer entirely and never unfreezes it after head
   warmup, so only the ~200K-parameter decision head (vs. the backbone's ~600M) is trained.
   Peak usage measured at ~2.8GB.
   ```bash
   python scripts/train_pipeline_decisions.py --input data/xiangqi_v3 --output-dir runs/xiangqi_full_v3 \
     --model Qwen/Qwen3-0.6B --objective gold_distribution --loss ce --set-head attention \
     --head-steps 60 --steps 1980 --batch-questions 4 --microbatch-questions 1 \
     --max-microbatch-tokens 4000 --eval-every 200 --precision bf16 --freeze-backbone
   ```
5. **Swap the checkpoint**: `serve_decisions.py --checkpoint-dir runs/xiangqi_full_v3` — the
   game server is unaware which checkpoint is behind port 8765.

### Measured results and what actually limits quality here

| Run | Data | Steps | Dev CE | Wall time |
|---|---:|---:|---:|---:|
| Baseline (untrained) | — | 0 | ~2.47-2.50 | — |
| `xiangqi_full_v1` | 1200 | 150 | 2.374 | ~2.0h |
| `xiangqi_full_v2` | 1200 | 430 | 2.355 | ~1.7h |
| `xiangqi_full_v3` | 6000 | 2040 | **2.053** | ~7.6h |

(Lower is better; CE = cross-entropy against Pikafish's chosen move. Test/OOD numbers are
omitted from this table because each run's eval split is a different freshly-generated set —
only Dev CE within a run, and across runs at the same eval-set size, is a fair comparison.)
Every run's *best* checkpoint (by dev CE) came from a step well before the final one — v1 at
150/150 (borderline), v2 at 370/430, v3 at 1460/2040 — later steps consistently drift back up,
the expected overfitting signature of a fixed-size pool. **v3's 5x larger dataset produced the
largest single improvement of the three runs**, reinforcing that data volume, not step count,
is the binding constraint here.

Two operational lessons learned running these:
- **Check `nvidia-smi` for competing processes before trusting any timing estimate on a small
  GPU.** Per-question cost swung between ~10.5s and ~3.5s across runs purely depending on
  whether the idle `serve_decisions.py`/`serve_xiangqi.py` processes were left resident on
  the same 4GB card during training.
- **`nvidia-smi`'s reported memory can look alarming (crept to ~3.8GB during v3) without
  reflecting real risk.** `torch.cuda.max_memory_allocated()` — the actual peak tensor
  allocation — stayed at 2.77GB across all three runs regardless of dataset size or batch
  size; the gap is the CUDA caching allocator holding reserved-but-unused blocks, which
  fluctuates with the specific token/candidate-count shapes a given batch happens to sample.
  `expandable_segments:True` (PyTorch's usual fix for this) is **not supported on Windows**;
  a periodic `torch.cuda.empty_cache()` every 20 steps was added to `train_pipeline_decisions.py`
  as a cross-platform mitigation instead.

Keep eval splits small (dev/calibration ~15-20, test/ood ~15-30 was used for v3): the
mandatory full-split evaluation that runs once after training scales directly with their
combined size, and at this GPU's per-question cost, large eval splits alone can eat hours.
