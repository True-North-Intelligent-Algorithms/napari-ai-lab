"""Run a scikit-ops op as a global segmenter.

The first step of docs/spec/0004. One skop op is registered explicitly and
appears in the segmenter list beside the hand-written segmenters, but runs in
scikit-ops' own environment rather than this one -- which is what lets the host
environment drop TensorFlow, micro-sam and friends, and move forward a napari
major version.

Deliberately minimal for this iteration:

* No parameters. The op runs on its defaults, so nothing here duplicates the
  op's signature. Generating the parameter form from that signature, via
  skop-napari and magicgui, is the next step and is the whole point -- writing
  the parameters out by hand here would be the opposite of it.
* No training. skop has no training ops yet (see 0001).
* Availability is reported as plain True. It deserves a third state -- "runs
  elsewhere, and the environment may still need building, which costs minutes
  and gigabytes" -- and that is coming as ``availability()``.
"""

from dataclasses import dataclass

from .GlobalSegmenterBase import GlobalSegmenterBase

try:
    from skop.runner import Runner

    _is_skop_available = True
except ImportError:  # optional dependency, see docs/spec/0003
    Runner = None
    _is_skop_available = False


#: One Runner for the process. It caches built environments, so sharing it
#: means the environment is located once rather than per segmentation.
_runner = None


def _get_runner():
    global _runner
    if _runner is None:
        _runner = Runner()
    return _runner


@dataclass
class SkopSegmenter(GlobalSegmenterBase):
    """Base for segmenters backed by a scikit-ops op.

    Not registered itself. ``register_op`` builds and registers one subclass
    per op, because the registry holds classes and instantiates them with no
    arguments.
    """

    #: The skop op this segmenter calls. Set on the subclass by register_op.
    op = None

    instructions = """
scikit-ops segmenter:
• Runs in scikit-ops' own environment, not this one
• The first run builds that environment, which takes minutes and gigabytes
• Later runs reuse it
• Parameters are not adjustable yet -- the op runs on its defaults
    """

    def __post_init__(self):
        # Set directly rather than through SegmenterBase.__init__, which a
        # dataclass's generated __init__ does not call.
        self._potential_axes = ["YX", "YXC"]
        self._supported_axes = ["YX", "YXC"]

    def are_dependencies_available(self) -> bool:
        """True when scikit-ops is importable here.

        Note what this does *not* claim: the op's own stack (TensorFlow, for
        StarDist) is not here and never will be. That is skop's problem, and
        the reason this returns True is that from ai-lab's point of view the
        only local dependency is skop itself.

        Returning False would send this down the execute_appose path in
        ImageDataModel.segment, which is the machinery scikit-ops replaces.
        """
        return _is_skop_available and self.op is not None

    def get_execution_string(self, image, **kwargs) -> str:
        """Part of the appose-registry paradigm that scikit-ops replaces.

        Retained only so this class satisfies the same contract as the other
        global segmenters. It goes when execute_appose.py does -- see 0001.
        """
        return ""

    def segment(self, image, **kwargs):
        """Run the op and return its label image.

        ``kwargs`` carries points, shapes and parent_directory from
        ImageDataModel.segment. None of them apply to a global op, so they are
        accepted and ignored, as the other global segmenters do.
        """
        if not self.are_dependencies_available():
            raise ImportError(
                "This segmenter requires scikit-ops, which is not installed. "
                "Install it with: pip install scikit-ops"
            )
        return _get_runner().run(self.op, image=image)

    @classmethod
    def register_op(cls, op, name: str, potential_axes=None):
        """Register one op under *name*, as its own segmenter subclass.

        Explicit rather than discovered: skop.discovery would be slow and
        would flood a list that is deliberately curated.
        """
        # staticmethod, not the bare function. A function stored as a class
        # attribute is a descriptor, so `self.op` would hand back a *bound
        # method* -- and skop then tries to attach its spec to that, failing
        # with "'method' object has no attribute '__skop_spec__'".
        namespace = {
            "op": staticmethod(op),
            "__doc__": f"scikit-ops op: {op.__name__}",
        }
        if potential_axes is not None:
            namespace["_default_potential_axes"] = potential_axes
        subclass = dataclass(type(name.replace(" ", "_"), (cls,), namespace))
        GlobalSegmenterBase.register_framework(name, subclass)
        return subclass
