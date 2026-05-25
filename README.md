# ABX for semantics and syntax

Used in the DP-Parse and tGSLM papers.

Problems:
- Features for words not contextualized
- Also train splits

<!-- griffe -->
## API reference

Syntactic and semantic ABX.

### `abx_pos`

```python
abx_pos(path_features, path_words, *, split='test', frequency=50, threshold=10, seed=0)
```

Compute the ABX part-of-speech score for the given features.

**Parameters:**

- **path_features** (<code>str | Path</code>) – Path to the directory containing feature ``.pt`` files.
- **path_words** (<code>str | Path</code>) – Path to the word annotations file.
- **split** (<code>Literal['dev', 'test']</code>) – Dataset split to evaluate on, either ``"dev"`` or ``"test"``.
- **frequency** (<code>int</code>) – Feature frequency used when loading data.
- **threshold** (<code>int</code>) – Maximum group size for subsampling.
- **seed** (<code>int</code>) – Random seed for subsampling.

**Returns:**

- <code>float</code> – ABX score (angular distance, lower is better).

### `abx_semantic`

```python
abx_semantic(path_features, path_words, *, split='test', frequency=50, threshold=10, seed=0)
```

Compute the ABX semantic score for the given features.

**Parameters:**

- **path_features** (<code>str | Path</code>) – Path to the directory containing feature ``.pt`` files.
- **path_words** (<code>str | Path</code>) – Path to the word annotations file.
- **split** (<code>Literal['dev', 'test']</code>) – Dataset split to evaluate on, either ``"dev"`` or ``"test"``.
- **frequency** (<code>int</code>) – Feature frequency used when loading data.
- **threshold** (<code>int</code>) – Maximum group size for subsampling.
- **seed** (<code>int</code>) – Random seed for subsampling.

**Returns:**

- <code>float</code> – ABX score (angular distance, lower is better).

### `read_triplets`

```python
read_triplets(task, split)
```

Load ABX triplets from the bundled asset file for the given task and split.

**Parameters:**

- **task** (<code>Literal['semantic', 'pos']</code>) – Evaluation type, either ``"pos"`` (part-of-speech) or ``"semantic"``.
- **split** (<code>Literal['dev', 'test']</code>) – Dataset split to load, either ``"dev"`` or ``"test"``.

**Returns:**

- <code>DataFrame</code> – DataFrame of ABX triplets with columns a, b (exploded), and x.

### `read_words`

```python
read_words(source)
```

Read word annotations from a space-separated file into a DataFrame.

**Parameters:**

- **source** (<code>str | Path</code>) – Path to the space-separated file with columns file, onset, offset, word.

**Returns:**

- <code>DataFrame</code> – DataFrame with columns file (str), onset (Decimal), offset (Decimal), word (str).


<!-- /griffe -->
