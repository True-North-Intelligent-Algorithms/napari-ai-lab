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
    from skop import Role, spec
    from skop.runner import Runner

    _is_skop_available = True
except ImportError:  # optional dependency, see docs/spec/0003
    Role = None
    spec = None
    Runner = None
    _is_skop_available = False

try:
    from skop_napari._roles import annotation_for, value_for
    from skop_napari._widget import build_inputs

    _is_skop_napari_available = True
except ImportError:  # the generated form is optional; defaults still work
    annotation_for = value_for = build_inputs = None
    _is_skop_napari_available = False


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
• Parameters below are generated from the op signature
    """

    def __post_init__(self):
        # Set directly rather than through SegmenterBase.__init__, which a
        # dataclass's generated __init__ does not call.
        self._potential_axes = ["YX", "YXC"]
        self._supported_axes = ["YX", "YXC"]
        #: The magicgui-built inputs, once a panel has been asked for. Holds
        #: the values the op will be called with.
        self._inputs = None

    def get_parameter_widgets(self):
        """Parameter widgets generated from the op's signature.

        The op signature already declares every parameter, its type, its
        default and its widget hints, so the form is generated from it rather
        than written out again as dataclass fields. NDOperationWidget lays
        these out in the same form as any other segmenter's, so the dependency
        banner, instructions and axis combo are unaffected.

        Returns an empty list when skop-napari is absent, which leaves the op
        on its defaults rather than failing.
        """
        if not (
            _is_skop_napari_available and self.are_dependencies_available()
        ):
            return []

        # Array inputs are skipped: napari-ai-lab chooses the image itself and
        # passes it to segment(). Offering a layer combo here would be a second
        # image selector disagreeing with the first.
        def skip(param):
            return param.role is Role.image

        self._inputs = build_inputs(
            spec(self.op), annotation_for, value_for, skip=skip
        )
        return self._inputs.widgets

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

    def availability(self):
        """Ready to run elsewhere, or ready once an environment is built.

        Never plain "available": the op does not run in this process, and the
        difference between an environment that exists and one that has to be
        built is minutes and gigabytes -- worth telling the user before they
        press the button rather than after.
        """
        from ...Segmenters.availability import ready, unavailable, will_build

        if not self.are_dependencies_available():
            return unavailable(
                "scikit-ops is not installed — pip install scikit-ops"
            )

        env_id = self._env_id()
        if self._environment_exists():
            return ready(
                f"Runs in the '{env_id}' environment, which is already built"
            )
        return will_build(
            f"Runs in the '{env_id}' environment, which must be built on "
            f"first use — this takes several minutes and gigabytes of disk"
        )

    def _environment_exists(self) -> bool:
        """Whether skop's environment for this op has already been built.

        Deliberately does not go through Runner.environment(), which *builds*
        the environment when it is missing -- asking whether a build is needed
        must not perform one. Instead it looks where appose keys them:
        <appose_envs_dir>/skop-<env_id>, matching Runner.environment's
        ``appose.pixi(config).name(f"skop-{env_id}")``.

        Any failure counts as not-built. Warning about a build that turns out
        to be instant is better than a silent multi-minute pause.
        """
        try:
            from pathlib import Path

            from appose.util.environment import appose_envs_dir

            return (
                Path(appose_envs_dir()) / f"skop-{self._env_id()}"
            ).is_dir()
        except Exception:  # noqa: BLE001 - informational only
            return False

    def _env_id(self) -> str:
        """The environment this op declares, or "unknown"."""
        try:
            return spec(self.op).env or "unknown"
        except Exception:  # noqa: BLE001 - informational only
            return "unknown"

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
        # Read the generated form at call time rather than tracking a
        # changed signal: the widgets already hold the current values, so
        # there is nothing to keep in sync. Empty when no panel was built,
        # which leaves the op on its own defaults.
        params = self._inputs.values() if self._inputs is not None else {}
        return _get_runner().run(self.op, image=image, **params)

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
