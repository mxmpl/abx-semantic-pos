"""Subsampler matching the original implementation's first-N selection.

The original (``replication/original/abx_data/abxeval_new.py``) selects
pronunciations via ``range(min(len(batch), thresh))`` — i.e. the first
``thresh`` rows of each word's per-pronunciation batch in npz order. The
default :class:`fastabx.Subsampler` shuffles per cell with a seed before
taking ``head``, which makes its choice depend on Polars' shuffle order
and the seed rather than on source order.

This module provides a drop-in replacement that takes ``head(size)`` of
each ``index_*`` list column without shuffling. Combined with an npz
whose per-word rows are in the same order as the source words DataFrame,
both implementations then consume the same pronunciations per word.
"""

import polars as pl
import polars.selectors as cs

__all__ = ["HeadSubsampler"]


class HeadSubsampler:
    """Subsampler matching the original implementation's deterministic head selection.

    Each cell's ``index_a`` / ``index_b`` / ``index_x`` list is truncated to
    its first ``max_size_group`` entries in source order. The ``seed`` and
    ``max_x_across`` arguments are accepted for API compatibility with
    :class:`fastabx.Subsampler` but unused.

    :param max_size_group: Maximum number of A, B, or X items per cell.
        Disabled if set to ``None``.
    :param max_x_across: Unused. Across-group subsampling is not performed.
    :param seed: Unused.
    """

    def __init__(self, max_size_group: int | None, max_x_across: int | None = None, seed: int = 0) -> None:
        """Create the subsampler."""
        self.max_size_group = max_size_group
        self.max_x_across = max_x_across
        self.seed = seed

    def __call__(self, lazy_cells: pl.LazyFrame, *, with_across: bool) -> pl.LazyFrame:
        """Truncate each cell's index lists to the first ``max_size_group`` items."""
        del with_across
        if self.max_size_group is None:
            return lazy_cells
        return lazy_cells.with_columns(cs.starts_with("index").list.head(self.max_size_group))

    def description(self, *, with_across: bool) -> str:
        """Return a description of the subsampling."""
        del with_across
        if self.max_size_group is None:
            return ""
        return f"maximal number of A, B, or X in a cell: {self.max_size_group} (head, no shuffle)"
