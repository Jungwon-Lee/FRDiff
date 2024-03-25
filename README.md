# FRDiff : Feature Reuse for Universal Training-free Acceleration of Diffusion Models [[Arxiv]](https://arxiv.org/abs/2312.03517)


## Overview 
In our work, we introduce an advanced acceleration technique that leverages the ___temporal redundancy___ inherent in diffusion models. Reusing feature maps with high temporal similarity opens up a new opportunity to save computation resources without compromising output quality. To realize the practical benefits of this intuition, we conduct an extensive analysis and propose a novel method, ___FRDiff___. ___FRDiff___ is designed to harness the advantages of both reduced NFE and feature reuse, achieving a Pareto frontier that balances fidelity and latency trade-offs in various generative tasks. 

<p align="center">
<img src=assets/main.png />
</p>

## Stable Diffusion

### Requirements
```
pip install torch transformers accelerate
pip install diffusers==0.26.3
```

### Sampling
To enjoy FRDiff on Stable Diffusion (SDXL), run following script.
```
cd Stable-Diffusion
python sample.py --num_steps 50 --interval 2 --prompt "a photo of an astronaut on a moon"
```

## DiT

### Requirements
```
conda env create --file DiT/environment.yml
```

### AutoFR Training & Sampling
To run AutoFR training, use following script.
```
cd DiT
python sample.py --lr 5e-3 --wgt 1e-3
```

The trained keyframeset and generated sample results will be saved in directory.  

To use Uniform Keyframeset, uncomment line 137-139 in DiT/model.py


## Citation
```
@article{so2023frdiff,
  title={FRDiff: Feature Reuse for Exquisite Zero-shot Acceleration of Diffusion Models},
  author={So, Junhyuk and Lee, Jungwon and Park, Eunhyeok},
  journal={arXiv preprint arXiv:2312.03517},
  year={2023}
}
```
