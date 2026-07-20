"""pytest collection-time guard against importing the real sentence-transformers stack.

WHY THIS FILE EXISTS
---------------------
``metrics/accuracy.py`` has a module-level::

    from sentence_transformers import SentenceTransformer, util

This line runs the instant ``metrics.accuracy`` is imported -- which happens at
*collection* time, when pytest imports ``test_accuracy_scorer.py`` and it does
``from metrics.accuracy import AccuracyScorer``. That import pulls in the
``sentence-transformers`` package, which transitively imports ``transformers``,
``huggingface_hub``, ``tokenizers``, etc. -- the actual source of the ~14.76s
slowdown.

That cost is paid *before* any fixture or ``@patch`` in the test file ever
runs. ``unittest.mock.patch("metrics.accuracy.SentenceTransformer", ...)`` is
a runtime substitution: it swaps an attribute on a module object that must
already exist in ``sys.modules``. It stops the real *model* from being
instantiated (no weights loaded), but it cannot stop the *import* from
happening, because the import already finished before the patch context
manager or fixture executed. So the existing autouse fixture in
``test_accuracy_scorer.py`` was doing exactly what it should -- it just
couldn't reach far enough back in time to matter.

THE FIX
-------
pytest always imports ``conftest.py`` before it collects (imports) any test
module in the same directory or below. By installing a lightweight fake
``sentence_transformers`` module into ``sys.modules`` here, we guarantee the
fake is already in place before ``test_accuracy_scorer.py``'s
``from metrics.accuracy import AccuracyScorer`` executes. ``metrics/accuracy.py``'s
top-level import then binds to this fake module instead of the real package,
and the real transformers/torch-adjacent import chain is never touched.

``util.cos_sim`` is reimplemented with real ``torch`` tensor ops (normalize +
matmul), matching the real sentence-transformers implementation, so cosine
similarity scores computed in tests stay numerically correct -- only the
heavy, unnecessary parts of the import graph are skipped.

This runs once per test session/module (whenever this conftest.py is first
imported by pytest), which is why it doesn't need explicit fixture wiring:
being present on disk in the right directory is what makes it fire at the
right time.
"""

from __future__ import annotations

import sys
import types

import torch


def _install_fake_sentence_transformers() -> None:
    if "sentence_transformers" in sys.modules:
        print("!!! STUB SKIPPED - sentence_transformers already loaded !!!")
        return
    print(">>> installing fake sentence_transformers stub")

    fake_module = types.ModuleType("sentence_transformers")

    class SentenceTransformer:  # noqa: N801 - mirrors real class name
        """Stand-in for the real class.

        Every test either patches ``metrics.accuracy.SentenceTransformer``
        directly or injects a mock via ``AccuracyScorer(model=...)``, so this
        constructor should never actually run. If it does, that signals a
        test is missing its mock -- fail loudly instead of silently trying
        to load a real model.
        """

        def __init__(self, model_name: str) -> None:
            raise RuntimeError(
                "SentenceTransformer() was instantiated with a real "
                f"model_name={model_name!r} during unit tests. A test is "
                "missing its mock/DI of the embedding model."
            )

    class util:  # noqa: N801 - mirrors real module name
        @staticmethod
        def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if a.dim() == 1:
                a = a.unsqueeze(0)
            if b.dim() == 1:
                b = b.unsqueeze(0)
            a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
            b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
            return torch.mm(a_norm, b_norm.transpose(0, 1))

    fake_module.SentenceTransformer = SentenceTransformer
    fake_module.util = util
    sys.modules["sentence_transformers"] = fake_module
    sys.modules["sentence_transformers.util"] = util


_install_fake_sentence_transformers()