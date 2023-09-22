from abc import ABC, abstractmethod


class OutputHandler(ABC):
    """Base class for model output handlers."""
    def __init__(self):
        pass

    def __call__(self, outputs):
        """Process outputs interface."""
        return self._process(outputs)

    @abstractmethod
    def _process(self, outputs):
        """Abstract postprocess method."""
        pass


class IdentityHandler(OutputHandler):
    """Identity postprocess class."""
    def _process(self, outputs):
        """Identity postprocess method.

        Args:
            outputs (Any):
                model outputs
        Returns:
            outputs (Any)
                unchanged model outputs
        """
        return outputs

