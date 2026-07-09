"""EDEN-7B activation harvesting for SAE training.

:mod:`~basecamp_eden_saes.harvest.eden` runs EDEN's partial forward to capture a
layer's residual stream; :mod:`~basecamp_eden_saes.harvest.streaming` produces
activations on the fly; :mod:`~basecamp_eden_saes.harvest.disk` pre-harvests them
to a goodfire-core activation store; and
:mod:`~basecamp_eden_saes.harvest.code_store` writes full-sparse SAE codes for the
annotation panel.
"""

from __future__ import annotations
