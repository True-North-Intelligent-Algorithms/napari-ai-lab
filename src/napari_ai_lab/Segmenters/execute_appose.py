"""
Appose Execution Utility.

This module provides utilities for executing segmentation remotely via appose
when dependencies are not available in the local environment.
"""


def execute_appose(image, segmenter, environment_path):
    """
    Execute segmentation remotely via appose.

    Args:
        image (numpy.ndarray): Input image to segment.
        segmenter: Segmenter instance with get_execution_string() method.
        environment_path (str): Path to the remote environment containing dependencies.

    Returns:
        appose.NDArray or None: Segmentation mask if successful, None if failed.

    Raises:
        ImportError: If appose is not available.
        RuntimeError: If remote execution fails.
    """
    try:
        import appose

        execution_string = segmenter.get_execution_string(image)

        env = appose.base(environment_path).build()
        ndarr_img = appose.NDArray(dtype=str(image.dtype), shape=image.shape)
        ndarr_img.ndarray()[:] = image

        with env.python() as python:
            task = python.task(
                execution_string, inputs={"image": ndarr_img}, queue="main"
            )
            task.wait_for()

            if task.error:
                print(f"⚠️  Task error: {task.error}")
                return None

            result = task.outputs.get("mask", None)
            return result

    except (ImportError, AttributeError, RuntimeError, OSError) as e:
        print(f"❌ Remote execution failed: {e}")
        return None
