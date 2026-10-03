"""Moved to ``kryptos_pii.candidates``.

The candidate proposer and the question LAYA answers are used by both training
and inference, and inference ships as a wheel. While this module lived under
``finetune/`` it could not travel with that wheel -- the directory also holds
the dataset, the training scripts and the checkpoint itself, none of which belong
in an installed package -- so ``kryptos-pii`` could not be installed anywhere
but a checkout.

The code now lives in the package that has to ship. This module makes itself an
alias for it, so ``import common`` and ``from common import pieces`` keep
working unchanged in every training script, and there is still exactly one
definition of how text is split. Aliasing the module object rather than
re-exporting its names also keeps ``common.set_context_chars(...)`` operating on
the same state the detector reads.
"""

import sys

from kryptos_pii import candidates

sys.modules[__name__] = candidates
