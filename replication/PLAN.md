# Replication script — plan

## Goal

`replication/main.py` must show that this library (`abx_semantic_pos`) produces
the same ABX scores as the original (`replication/original/abx_data/abxeval_new.py`)
on the **same** per-pronunciation embeddings, exercising both implementations
through their *standard* entry points.

The current `main.py` is rejected because:

- It hardcodes too much logic that already exists in the original
  (npz construction, task-file iteration, threshold handling).
- It does not actually reproduce the original's subsampling. It pre-shuffles
  the npz to *imitate* the new library's `Subsampler` — that's the wrong
  direction (we want the new lib to match the original, not vice-versa) and
  it leaves a residual ~3.6e-3 gap on the semantic splits.
- It conflates "features" with "writing tensors": the features source should
  be pluggable so a future run with real model embeddings only swaps one
  function.

## Spec, restated

1. **True words.** Use one LibriSpeech split's annotation file from
   `replication/original/librispeech/words/<subset>` verbatim. No custom
   alignment, no concatenation across subsets, no padding rows.
2. **Random features by default, real features pluggable.** The script must
   expose a single feature-provider function (`words_df -> np.ndarray[N, D]`).
   The default implementation returns deterministic random vectors; a real
   extractor (e.g. wrapping `preextracted_lm_tasks.get_features`) can be
   dropped in without touching the rest of the script.
3. **Standard workflows of both implementations.**
   - Original: build the per-word npz the way the original *itself* does (one
     value per word = stack of N pronunciation vectors), then call
     `abxeval_new.evaluate_similarity_task` on each `(syn|pos)_{dev|test}.txt`
     under `replication/original/abx_data/`.
   - New library: write per-file `.pt` feature tensors at the given frequency,
     then call `abx_semantic` / `abx_pos` with `path_features` + `path_words`.
4. **Identical subsampling.** Both sides must consume the exact same subset
   of pronunciations per word so the scores can be compared at machine
   precision (up to the original's `np.around(_, 2)` quantization, ~1e-4).

## Standard workflow of each implementation

### Original (`abxeval_new.evaluate_similarity_task`)

- Input: task file (`a x b1 b2 ...` lines) + `features: dict[word] -> (N, D)`.
- Per `(a, x)` row: `A_batch = features[a]`, `X_batch = features[x]`, then
  for each `b` in row: `B_batch = features[b]`. Iterates `a in range(min(len(A_batch), thresh))`,
  `x in range(min(len(X_batch), thresh))`, `b in range(min(len(B_batch), thresh))`.
  → **Takes the first `thresh=10` rows of each batch in npz order.**
- Score: per `(a_word, x_word, b_word)` key, mean over the `a×x×b` triples
  (1 if `cos(A,X) > cos(B,X)` else 0; ties counted as 0). Final score = mean
  over keys, then `np.around(100 * mean, 2)`.
- Npz construction (see `preextracted_lm_tasks.get_features`): build
  `type_list: dict[word, list[(1, D) np.ndarray]]` by appending one row per
  encountered pronunciation, then `np.concatenate(axis=0)` and
  `np.savez(output, **type_list)`. The docstring at `abxeval_new.py:99-102`
  says all entries should have the same `N`, but the code only requires
  `N >= 1` (it clips to `thresh`).

### New library (`abx_semantic` / `abx_pos`)

- Input: directory of per-file `.pt` tensors (one tensor per LibriSpeech
  recording, shape `[T, D]`, at `frequency` Hz) + word annotation file.
- Reads words → `_build_cells_and_labels` (`src/abx_semantic_pos/_core.py`):
  joins triplets against `(word -> [source_row_indices])`, applies the
  subsampler per cell, then filters rows used and remaps indices.
- Subsampling is now provided by the library's own
  `HeadSubsampler` (`src/abx_semantic_pos/_subsampler.py`), wired into
  `_core.py` in place of `fastabx.Subsampler`. It applies
  `cs.starts_with("index").list.head(size)` — **no shuffle, no seed
  dependency** — i.e. it keeps the first `threshold` source-row indices per
  cell, matching the original's `range(min(N, thresh))`.
- Mean-pools each pronunciation's frame window and runs
  `Task(...).is_symmetric = False`, then `Score(task, "angular").collapse(levels=["b"])`.

## Subsampling — now matched at the library level

Resolved. `HeadSubsampler` deterministically truncates each cell's
`index_a` / `index_b` / `index_x` lists to their first `max_size_group`
entries in source order. When the npz is built so that
`npz[word]` lists pronunciations in source-row order, both implementations
consume the same first `thresh` rows for every word.

The `seed` argument on `abx_pos` / `abx_semantic` is now unused (forwarded
to `HeadSubsampler`, which ignores it). Kept for now; cleanup later.

The previously-considered alternatives (pre-shuffling the npz to match
`fastabx.Subsampler`, or truncating words to ≤ threshold at the source) are
no longer needed.

## Steps

1. **Lift original npz construction into a small helper.** Reference:
   `preextracted_lm_tasks.get_features`. Minimal equivalent: walk the words
   DataFrame in row order, group by word, stack per-row feature vectors. Save
   via the `zipfile`-based `_save_npz` shim (`np.savez(file=..., **arrays)`
   collides with the literal word `"file"` in the vocab — keep the
   workaround, isolated in one helper).
2. **Extract a `provide_features(words_df) -> np.ndarray[N, D]` interface.**
   Default: `np.random.default_rng(SEED).standard_normal((len(words), DIM))`.
   Document the contract: one vector per `words_df` row, in row order. Leave
   a `# TODO: real feature provider` stub.
3. **Write per-file `.pt` features for the new lib.** Reuse the existing
   `_write_features` logic (fills each annotation's `[start, end)` frame
   window with that row's vector so mean-pooling reproduces the per-row
   vector).
4. **Build the npz in source-row order.** Group the words DataFrame by word
   *with `maintain_order=True`*, stack the per-row vectors, save. This makes
   `npz[word][:thresh]` identical to "first `thresh` rows for that word in
   the words file" — exactly what `HeadSubsampler` selects in cells.
5. **Call each implementation through its standard entry point.**
   - Original: `original.evaluate_similarity_task(task_txt,
     original.load_feature(npz_path), 0, task)`. For the task file, filter
     the bundled `replication/original/abx_data/{pos,syn}_{dev,test}.txt` to
     triplets whose A/B/X are all attested in the chosen subset (the new lib
     drops the others via inner-join, so this keeps the two sides on the
     same triplet set). Keep `_write_filtered_task`.
   - New lib: `abx_semantic(feat_dir, words_path, split=split,
     frequency=FREQUENCY, threshold=THRESHOLD, seed=SEED)` and same for
     `abx_pos`. `path_words` is the chosen LibriSpeech annotation file,
     verbatim.
6. **Compare with `TOL=5e-4`.** With identical pronunciations on both sides
   and random vectors (ties statistically zero), the only remaining
   difference is the original's `np.around(_, 2)` percentage quantization
   (~1e-4). Tighten `TOL` to `1e-3` after first verification.

## Difficulties / open items

1. **Triplet/vocab coverage on a single subset.** A single split (e.g.
   `dev-clean`) covers only part of the bundled triplets' vocabulary. The
   new lib drops uncovered triplets via inner-join on `word`; the original
   must be fed a filtered task file to drop the same ones. `_write_filtered_task`
   already does this — verify after the rewrite that both sides keep the
   same triplet set.
2. **`np.savez` kwarg collision.** The vocab contains the literal word
   `"file"`, which collides with `np.savez(file=..., **arrays)`. Keep the
   `zipfile`-based `_save_npz` helper with a comment pointing at the
   collision.
3. **`multiprocessing` start method.** The original uses `Pool`; on macOS we
   must `set_start_method("fork", force=True)` before any pool is created
   because the module is loaded by path via `spec_from_file_location`.
   Keep at the top of `main.py`.
4. **Averaging convention.** Already addressed: `Score(task, "angular").collapse(levels=["b"])`
   matches the original's `mean over (a,x,b) keys`. Re-verify: if a residual
   gap survives `HeadSubsampler`, averaging is the next suspect (e.g.
   whether the new lib weights cells differently when `len(b_words)` varies
   across `(a, x)` rows).
5. **Ties.** Original counts ties as 0 (`>`), new lib counts ties as 0.5.
   With random Gaussian vectors at `DIM=32`, ties don't occur in practice.
   Document but don't fix.
6. **Real-features extension.** Real extraction in `preextracted_lm_tasks.py`
   relies on Fairseq + GPU + kmeans + LM checkpoints. The script should
   leave a clearly marked extension point and not attempt to wire that path
   itself. The contract for `provide_features` (one vector per `words_df`
   row, in row order) is enough to plug in real features later.
7. **Unused `seed` argument.** `abx_pos` / `abx_semantic` still accept `seed`
   but `HeadSubsampler` ignores it. Cleanup deferred; flag for a later pass.

## Out of scope

- Modifying `abxeval_new.py`.
- Synthesizing or padding word annotations.
- Running the original's `main()` (per-layer directory walking) — we call
  `evaluate_similarity_task` directly.
- Wiring up the real-features path. Plan documents the extension point only.
- Removing the now-unused `seed` parameter from the library's public API.
