# FRDiff with Stable Diffusion

This code is for FRdiff with Stable Diffusion (SDXL).

## Requirements
```
pip install torch transformers accelerate
pip install diffusers==0.26.3
```

## Sampling
```
python sample.py --num_steps 50 --interval 2 --prompt "a photo of an astronaut on a moon"
```
