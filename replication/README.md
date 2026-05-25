# Replication

Original implementation in `https://gitlab.cognitive-ml.fr/ralgayres/abx_sem_syn`.

Files:
- Words: `./original/librispeech/words`
- POS and semantic annotations: `./original/abx_data/{task}_{split}.txt` with `task` either "pos" or "syn" and `split` either "dev" or "test".
- ABX computation: `./original/abx_data/abxeval_new.py`

The feature extraction is likely also in `./original/abx_data`.

Note: the task is named "syn" but it corresponds to the semantic ABX, not the syntactic (this one is "pos", for part-of-speech).
