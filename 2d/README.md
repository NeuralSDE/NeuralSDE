<h1 align="center">Neural SDEs as a Unified Approach to Continuous-Domain Sequence Modeling</h1>
<p align="center">
  <h3 align="center">Under Double Blind Review</h3>
</p>

This directory contains the implementation for the the 2D Bifurcation task. Our approach is able to capture multi-modal distributions (2 branches).

## 🚀 Quick Start

### Setup

Create a conda environment and install dependencies:
```bash
cd 2d
conda env create -f environment.yml
conda activate 2d
```
### Training
We have provided checkpoints for this task, so you can directly proceed to evaluation.

We separately train flow, denoiser, and diffusion components. The diffusion component is based on a specific checkpoint of the flow model.

To train a flow model:
1. Set `train_flow: true` in [configs/branching-100.yaml](configs/branching-100.yaml)
2. Set `train_denoiser: false` and `train_diffusion: false` in the same config file
3. Execute the following command:
```bash
python trainer.py --config ./configs/branching.yaml --training
```
Your training runs will be saved under the `./runs` directory.

To train a denoiser:
1. Set `train_denoiser: true` in [configs/branching-100.yaml](configs/branching-100.yaml)
2. Set `train_flow: false` and `train_diffusion: false` in the same config file
3. Execute the following command:
```bash
python trainer.py --config ./configs/branching.yaml --training
```

To train a diffusion model:
1. Set `train_diffusion: true` in [configs/branching-100.yaml](configs/branching-100.yaml)
2. Set `train_flow: false` and `train_denoiser: false` in the same config file
3. Specify the path to your trained flow model in `flow_path` under the `diffusion` section
4. Execute the following command:
```bash
python trainer.py --config ./configs/branching.yaml --training
```

### Evaluation
To evaluate the models:
1. Set the following paths in [configs/branching-100.yaml](configs/branching-100.yaml):
   - `eval:flow_path`: Path to your trained flow model
   - `eval:denoiser_path`: Path to your trained denoiser model
   - `eval:diffusion_path`: Path to your trained diffusion model
2. Execute the following command:
```bash
python trainer.py --config ./configs/branching.yaml
```
The evaluation results will be saved under `./runs/eval`.
