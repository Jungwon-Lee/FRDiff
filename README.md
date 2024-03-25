# FRDiff : Feature Reuse for Universal Training-free Acceleration of Diffusion Models [[Arxiv]](https://arxiv.org/abs/2312.03517)

![main_small](https://github.com/Jungwon-Lee/FRDiff/assets/33821003/63374dc2-a530-4898-a12c-5761c0513959)


## Abstract 
The substantial computational costs of diffusion models, especially due to the repeated denoising steps necessary for high-quality image generation, present a major obstacle to their widespread adoption. While several studies have attempted to address this issue by reducing the number of score function evaluations (NFE) using advanced ODE solvers without fine-tuning, the decreased number of denoising iterations misses the opportunity to update fine details, resulting in noticeable quality degradation. In our work, we introduce an advanced acceleration technique that leverages the temporal redundancy inherent in diffusion models. Reusing feature maps with high temporal similarity opens up a new opportunity to save computation resources without compromising output quality. To realize the practical benefits of this intuition, we conduct an extensive analysis and propose a novel method, FRDiff. FRDiff is designed to harness the advantages of both reduced NFE and feature reuse, achieving a Pareto frontier that balances fidelity and latency trade-offs in various generative tasks.

## Motivation
In this paper, our primarily focus on the temporal behavior of diffusion model, stemming from their repeated denoising operation. Specifically, we observe that the temporal changes of diffusion model remains relatively small across most time steps, regardless of the architecture and dataset. 

![feature_map](https://github.com/Jungwon-Lee/FRDiff/assets/33821003/8a257f2e-1c15-4dbe-8806-fdc5e69b4b75)

## Overview
<img width="963" alt="method_temp1" src="https://github.com/Jungwon-Lee/FRDiff/assets/33821003/a5df7567-0611-4839-9335-85fb27667d49">


## Citation
```
@article{so2023frdiff,
  title={FRDiff: Feature Reuse for Exquisite Zero-shot Acceleration of Diffusion Models},
  author={So, Junhyuk and Lee, Jungwon and Park, Eunhyeok},
  journal={arXiv preprint arXiv:2312.03517},
  year={2023}
}
```
