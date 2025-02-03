# Installing ClimatExML

ClimatExML should be built from source so that users can modify the training to suit their needs or preferences. There are two main ways of using ClimatExML. 

1. (Recommended) By installing it and the requirements locally on your host machine or;
2. By using the containers that are designed to run this code. 

Option (1) requires slightly more configuration, but can be simpler to get started quickly, while option (2) is highly portable and provides convenient ways to configure the pipeline on HPC systems. Instructions are provided here to get started with both. 


```{note}
If installing locally, it is highly recommend that you use [Python virtual environments](https://docs.python.org/3/library/venv.html). Basic instructions are included below.
```
## Installation in a Virtual Environment
```
python -m venv climatexvenv
source climatexvenv/bin/activate
```

Begin by cloning [the ClimatExML repo](https://github.com/nannau/ClimatExML)

```
git clone https://github.com/nannau/ClimatExML 
# or
git clone git clone git@github.com:nannau/ClimatExML.git
```

Then install the ClimatExML Python package with 
```bash
pip install -e ClimatExML/
```

This should install the necessary requirements as well. 

## Verify NVidia GPU Hardware
Make sure the GPU is accessible and working correctly with

```
nvidia-smi
```

Which should return something like 

```
Tue Jan  9 14:30:42 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX XXXX        Off | 00000000:01:00.0  On |                  Off |
|  0%   35C    P8              21W / 450W |     61MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2020      G   /usr/lib/xorg/Xorg                           35MiB |
|    0   N/A  N/A      2079      G   /usr/bin/gnome-shell                         13MiB |
+---------------------------------------------------------------------------------------+
```

## Verify PyTorch + GPU Access

It's important to also verify that PyTorch is installed correctly and is functioning as expected. That is, PyTorch should be able to communicate with the GPU and load tensors on etc. Fortunately, this is easy to check with
```python
import torch
torch.cuda.is_available()
```

This should return `True`.

As an extra step, load a tensor onto the GPU with

```python
import torch
torch.randn(1000).cuda()
```

If no errors are raised then PyTorch is operating correctly.

If you encounter troubles, make sure you are using your virtual environment and that no other PyTorch versions are installed on your base machine as root (outside of your virtual environment).