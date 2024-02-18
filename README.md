# Qtransformer
 quantum transformer

# Installation
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
