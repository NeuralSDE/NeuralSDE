<h1 align="center">Neural SDEs as a Unified Approach to Continuous-Domain Sequence Modeling</h1>
<p align="center">
  <h3 align="center">Under Double Blind Review</h3>
</p>

This is the official repository for the paper [Neural SDEs as a Unified Approach to Continuous-Domain Sequence Modeling](). We introduce a novel approach to continuous sequence modeling that interprets timeseries data as discrete samples from an underlying continuous dynamical system. Our method models time evolution using Neural Stochastic Differential Equations (Neural SDEs), where both drift and diffusion terms are parameterized by neural networks. We provide a principled maximum likelihood objective and a simulation-free scheme for efficient training. Through extensive experiments across both embodied and generative AI tasks, we demonstrate that our SDE-based continuous-time modeling excels in complex, high-dimensional, and temporally intricate domains. This repository contains the following domains:

- **2D Branching Trajectories**: a 2D Bifurcation task designed to assess multi-modaltrajectory generation
- **Video Prediction**: video prediction on standard benchmark datasets
- **PushT**: a imitation learning task.

![project_teaser](/assets/overview_figure.jpg)

## 🔥 News
- **2025-03**: Initial release of Neural SDEs for continous sequence modeling!

## 📋 Todo 
- code for pushT
- pretrained model for video prediction

## 🔍 Project Structure

This project is organized into three main components, each with its own dedicated directory and documentation. For detailed instructions on each component, please refer to their respective README files. Each component has its own environment and dependencies.

1. Clone this repository:
```bash
git clone https://github.com/username/NeuralSDE.git
cd NeuralSDE
```

2. Set up the environment for your specific task:
   - For the 2D Bifurcation task, see [2D README](/2d/README.md)
   - For the video prediction task, see [Video Predction README](/video/README.md)
   - For the Push-T imitation learning task, see [Push-T README](/pushT/README.md)

##

## 📝 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@article{author2025neuralsde,
  title={Neural SDEs as a Unified Approach to Continuous-Domain Sequence Modeling},
  author={Author, A. and Author, B. and Author, C.},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## 🙏 Acknowledgements

This project builds upon and incorporates code from several open-source repositories. We would like to express our sincere gratitude to:

- [river](https://github.com/Araachie/river)
- [diffusion policy](https://github.com/real-stanford/diffusion_policy)

We also thank all colleagues and reviewers who provided valuable feedback during the development of this work.
