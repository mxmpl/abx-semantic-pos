# Replication script — WIP

## Goal

`replication/main.py` checks that `abx_semantic_pos` (this library) produces the same ABX scores
as the original implementation at `replication/original/abx_data/abxeval_new.py`. Both implementations
should consume the **same** per-pronunciation embeddings — the original via a per-word `npz` dict,
the new library via time-indexed `.pt` features + a `read_words`-compatible alignment file.

## Latest user constraints (in order received)

1. Use real LibriSpeech word annotations from `replication/original/librispeech/words/<subset>`, not
   synthesized ones.
2. Use the library's `read_words` function.
3. **Do not write a custom words file** — point the new lib's `path_words` at one of the original
   librispeech annotation files directly.
4. Use a **single subset** (e.g. `dev-clean`), not a glob/pool of subsets.

## Current state of `replication/main.py`

- Loads `replication/original/librispeech/words/dev-clean` via `read_words`.
- Generates one random float32 vector per annotation row (`DIM=32`, `SEED=1234`).
- Writes per-file `.pt` tensors to a `TemporaryDirectory`. Each annotation's `[start, end)` frame
  window (from `fastabx.dataset.item_frontiers`) is filled with that row's vector, so
  `mean`-pooling reproduces the per-pronunciation vector.
- Builds a per-word npz dict from the same vectors. Pre-arranges each word's vector list to match
  what the new library's `Subsampler` picks — `pl.Series(rows).shuffle(seed=SEED).to_list()` — so
  the original's `range(min(N, thresh))` and the new lib's first-`threshold` post-shuffle items
  reference the **same** vectors.
- Custom `_save_npz` writes the npz via `zipfile` to avoid `np.savez(file=...)` kwarg collision
  (literal triplet word `"file"` exists in librispeech vocab).
- For each `(task, split)`, filters the bundled triplets to those where all of A/B/X appear in the
  subset's vocab, regroups into `a x b1 b2 ...` lines, writes a filtered task file to `tmp_root`,
  and runs the original on it. The new library's inner-join on `word` drops the same triplets.
- Passes `replication/original/librispeech/words/dev-clean` verbatim as the new lib's `path_words`.

## What's working

- Both implementations run end-to-end on the same generated features + filtered triplets.
- All triplets dropped by the original (filtered task file) are also dropped by the new lib (inner
  join on `word`).
- The npz can be written despite the literal `"file"` word in the vocab.

## What's blocking exact matching

Final |Δ| with the npz aligned to the Subsampler shuffle and `TOL=5e-4`:

```
task      split     orig_err     new_err         |Δ|
----------------------------------------------------
semantic  dev       0.490800    0.494461    3.66e-03
semantic  test      0.500600    0.497029    3.57e-03
pos       dev       0.501000    0.501174    1.74e-04
pos       test      0.497000    0.498162    1.16e-03
```

I expected the alignment to drive these to ~1e-4 (the original's `np.around(_, 2)` quantization
floor on the percentage), but two splits stayed near 3.6e-3.

### Verified upstream of the run

- `pl.Series(indices).shuffle(seed=SEED).to_list()[:10]` matches the Subsampler's per-cell output
  bit-for-bit (verified with a 1-cell and a 4-cell DataFrame across `index_a`/`index_b`/`index_x`
  and with values that look like real librispeech row indices).
- The shuffle is partition-independent: all cells with `a="great"` get the same 10 picks.
- The averaging convention in the library was already updated earlier in this branch to
  `Score(task, "angular").collapse(levels=["b"])` to match the original's nested mean
  `mean_{(a,x)} mean_b mean_pronunciations`.
- Original applies `np.around(100 * mean, 2)` to its accuracy → quantizes its comparable error
  rate to a 1e-4 grid. That alone cannot explain a 3.6e-3 gap.

### Open questions / leads to investigate next session

1. **Where the residual gap comes from.** The alignment proof checks out in isolation, but the
   real run still diverges on `semantic/dev` and `semantic/test` at ~3.6e-3. Hypotheses:
   - The new lib's `_build_cells_and_labels` re-indexes `words` (filters to rows used, rebuilds
     row indices, remaps via `replace_strict`). After remap, the shuffle order seen by the
     downstream task may no longer be the original-row order — i.e. the alignment should be on
     the **remapped** index space, not the source-row space.
   - The Subsampler is invoked on `cells.lazy()` from `_build_cells_and_labels`, which has joined
     `triplets` against the per-word `idx` agg. The list values in `index_a`/`index_b`/`index_x`
     are the **source-words-df** row indices. Confirmed in my standalone tests this is what gets
     shuffled. So the shuffle output is in source-row space — the alignment should be correct.
   - But the per-pronunciation lookup at scoring time uses **remapped** indices into the filtered
     `labels` dataframe (and into the stacked `data` tensor). The mapping is built from
     `idx_used.sort("index")`. So vectors get permuted by this filtering+remapping, **independent
     of the npz order**. Need to verify whether my "first 10 of shuffled source-row list" actually
     corresponds to the first 10 vectors used at scoring time, or whether `idx_used.sort("index")`
     reorders them.
2. **Confirm cells contain the same (a,x,b) set in both impls.** For the splits with larger Δ,
   `semantic/dev` and `semantic/test`, dump the per-(a,x,b) score from each side and diff them.
   That will localize whether the issue is (a) different vectors used per cell, (b) different
   cells, or (c) averaging.
3. **Sanity: number of pronunciations per triplet word in `dev-clean`.** Max 96, median 2; 94
   words out of 1146 covered have >10 pronunciations. So only those 94 words exercise the
   subsampling-divergence path. If the gap is concentrated on triplets involving these 94 words,
   the alignment is the right fix and we just need to wire it through the remap.
4. **Alternative path: skip alignment, set THRESHOLD high enough that nobody subsamples.** Blocked
   by the original hard-coding `thresh=10` inside `evaluate_similarity_task`. Would require
   monkey-patching the original or modifying its source — the user previously asked for minimal
   modifications to the original.

## Constraints to respect next session

- Do **not** write a custom word-alignment file. The new lib's `path_words` must point at an
  existing librispeech annotations file (e.g. `dev-clean`).
- Use a **single** subset. No globs, no concatenation across subsets.
- Don't modify the original `abxeval_new.py`.
- Keep `multiprocessing.set_start_method("fork", force=True)` at the top of the script — the
  original uses `Pool` and spawn doesn't work because the module is loaded by file path.
- `tmp_root` should remain a `tempfile.TemporaryDirectory` context manager (per earlier request).

## Key references

- Library averaging convention: `src/abx_semantic_pos/_core.py` line 161-163 (`Task` +
  `collapse(levels=["b"])`).
- Library cells construction: `src/abx_semantic_pos/_core.py` `_build_cells_and_labels`. Note the
  `idx_used.sort("index")` → `with_row_index()` → `replace_strict` remap chain.
- Subsampler shuffle: `fastabx/subsample.py` line 9 `subsample_each_cell` —
  `cs.starts_with("index").explode().shuffle(seed=seed).implode().over("__group").list.head(size)`.
- Original's hardcoded threshold: `replication/original/abx_data/abxeval_new.py` line 64
  (`thresh=10` inside `evaluate_similarity_task`).
- Frame frontier math: `fastabx/dataset.py` `item_frontiers` —
  `start = ceil(onset*freq - 0.5)`, `end = floor(offset*freq - 0.5) + 1`.

## Earlier-state snapshots (for context)

- **Synthetic-features approach** (worked, |Δ| < 1e-4 across all 4 task/splits): one-frame-per
  -pronunciation `.pt` plus a hand-written `words.txt`. User rejected: they want real
  librispeech annotations.
- **Multi-subset pooled approach with synthetic padding** for 151 missing words (5 subsets:
  dev-clean, dev-other, test-clean, test-other, train-clean-100; |Δ| up to 1.07e-4): user
  rejected — wants a single subset, no padding writing.
