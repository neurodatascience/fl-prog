"""SCAFFOLD client module that leaves local (non-aggregated) parameters uncorrected.

Fed-BioMed strips `local`-tagged parameters from the auxiliary variables sent to
the researcher and refills them with zeros on the way back. declearn's
`ScaffoldClientModule` then computes `delta = local_state - 0` for those
parameters, i.e. it corrects their gradients by the node's own previous-round
mean gradient. Local parameters are not aggregated, so the SCAFFOLD correction
for them should be zero: this module masks it out.
"""

from collections.abc import Iterable

from declearn.model.api import Vector
from declearn.optimizer.modules import ScaffoldClientModule


class LocalAwareScaffoldClientModule(ScaffoldClientModule):
    name = "scaffold-client-local-aware"

    def __init__(self, local_names: Iterable[str] = ()) -> None:
        super().__init__()
        self.local_names = list(local_names)

    def get_config(self) -> dict:
        return {"local_names": self.local_names}

    def _masked_delta(self):
        delta = self.delta
        if not isinstance(delta, Vector) or not self.local_names:
            return delta
        coefs = dict(delta.coefs)
        for name in self.local_names:
            if name in coefs:
                coefs[name] = coefs[name] * 0
        return type(delta)(coefs)

    def run(self, gradients: Vector) -> Vector:
        # same as ScaffoldClientModule.run, with zero correction for local params
        self._grads = self._grads + gradients
        self._steps += 1
        return gradients - self._masked_delta()
