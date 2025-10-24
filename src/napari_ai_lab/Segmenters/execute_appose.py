"""
Appose Execution Utility.

This module provides utilities for executing segmentation remotely via appose
when dependencies are not available in the local environment.
"""


def execute_appose(image, segmenter, environment_path, additional_inputs=None):
    """
    Execute segmentation remotely via appose.

    Args:
        image (numpy.ndarray): Input image to segment.
        segmenter: Segmenter instance with get_execution_string() method.
        environment_path (str): Path to the remote environment containing dependencies.
        additional_inputs (dict, optional): Additional inputs to pass to the remote execution.

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

        # Always include image, optionally add additional inputs
        inputs = {"image": ndarr_img}
        if additional_inputs is not None:
            inputs.update(additional_inputs)

        with env.python() as python:
            task = python.task(execution_string, inputs=inputs, queue="main")
            task.wait_for()

            if task.error:
                print(f"⚠️  Task error: {task.error}")
                return None

            print(task.outputs.keys())

            print(task.outputs.get("test_list", None))

            result = task.outputs.get("mask", None)
            return result

    except (ImportError, AttributeError, RuntimeError, OSError) as e:
        print(f"❌ Remote execution failed: {e}")
        return None
