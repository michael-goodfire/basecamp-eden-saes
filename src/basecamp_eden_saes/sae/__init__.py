"""BatchTopK SAE construction, sweep training, selection, and quality analysis.

The shared-forward multi-SAE trainers live in
:mod:`~basecamp_eden_saes.sae.trainer` and
:mod:`~basecamp_eden_saes.sae.multitrainer`;
:mod:`~basecamp_eden_saes.sae.sweep` is the DDP grid entrypoint;
:mod:`~basecamp_eden_saes.sae.select` /
:mod:`~basecamp_eden_saes.sae.exemplars` / :mod:`~basecamp_eden_saes.sae.rank` /
:mod:`~basecamp_eden_saes.sae.quality` are the post-hoc analysis tools.
"""

from __future__ import annotations
