# Qtransformer
 quantum transformer

# Installation
Qtransformer works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 10.2+ and PyTorch 1.8+.

**Step 1** Create a conda environment and activate it.
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

```bash
git clone https://github.com/itachixc/qtransformer.git
cd qtransformer
pip install -U openmim && mim install -e .
```

# Verify the installation

```bash
python demo/image_demo.py demo/demo.JPEG resnet18_8xb32_in1k --device cpu
```
# Train

```bash
python tools/trains.py qtransformer_configs/vit_config_cifar10.py
```
