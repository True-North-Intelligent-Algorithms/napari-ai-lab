# Troubleshooting

An NVIDIA GPU is recommended. Things run on CPU, slowly.

- JupyterLab Kernel menu: Shut Down All Kernels (frees GPU)
- `pkill -f "share/appose/skop-"` kill leftover scikit-ops workers
