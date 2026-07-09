"""SAE model construction.

The canonical BatchTopK SAE constructor with the EDEN normalizer pipeline baked
in lives in :mod:`basecamp_eden_saes.harvest.streaming` (next to the streaming
producer that fits its normalizer stats). It is re-exported here so callers can
import it from the ``sae`` package. The underlying ``BatchTopKSAE`` itself comes
from ``goodfire_core.saes.batch_topk`` and is never reimplemented.
"""

from __future__ import annotations

from goodfire_core.saes.batch_topk import BatchTopKSAE

from basecamp_eden_saes.harvest.streaming import build_sae

__all__ = ["BatchTopKSAE", "build_sae"]
