# Quantum Vision Transformer
 Quantum vision transformer is a quantum-classical hybrid model that accelerates the vision transformer with quantum computing. 

 <div align="center">
<img src="https://github.com/itachixc/qtransformer/blob/main/docs/images/qvit.png"/>
</div>

# Installation
Qtransformer works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 10.2+ and PyTorch 1.8+.

**Step 1.** Create a conda environment and activate it.
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.**  Install PyTorch following official instructions, e.g. 
On GPU platforms:
```bash
conda install pytorch torchvision -c pytorch
```
On CPU platforms:
```bash
conda install pytorch torchvision cpuonly -c pytorch
```
**Step 3.**  Install qtransformer developed by mmpretrain
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
