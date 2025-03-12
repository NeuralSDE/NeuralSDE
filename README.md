<h1 align="center">Neural SDE: Stochastic Differential Equations for Generative Modeling</h1>
<p align="center">
  <h3 align="center">A Framework for Trajectory, Video, and Robot Action Generation</h3>
</p>

This is the official repository for the paper [Neural SDEs as a Unified Approach to Continuous-Domain Sequence Modeling](https://arxiv.org/pdf/2501.18871). We introduce a novel approach to continuous sequence modeling that interprets timeseries data as discrete samples from an underlying continuous dynamical system. Our method models time evolution using Neural Stochastic Differential Equations (Neural SDEs), where both drift and diffusion terms are parameterized by neural networks. We provide a principled maximum likelihood objective and a simulation-free scheme for efficient training. Through extensive experiments across both embodied and generative AI tasks, we demonstrate that our SDE-based continuous-time modeling excels in complex, high-dimensional, and temporally intricate domains. This repository contains the following domains:

- **2D Branching Trajectories**: Generate complex 2D trajectories with precise control over dynamics
- **Video Prediction**: Create temporally consistent videos with controllable motion
- **Imitation Learning (PushT)**: Synthesize physically plausible robot action sequences

![project_teaser](path/to/teaser.png)

## 🔥 News
- **2025-03**: Initial release of Neural SDE framework

## 📋 Todo 
code for pushT
code for video prediction

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
@article{author2024neuralsde,
  title={Neural SDEs as a Unified Approach to Continuous-Domain Sequence Modeling},
  author={Author, A. and Author, B. and Author, C.},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## 📄 License

This project is licensed under the [LICENSE](LICENSE) file included in the repository.

## 🙏 Acknowledgements

We thank [list of people or organizations] for their valuable contributions and support.
