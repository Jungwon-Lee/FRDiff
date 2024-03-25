# FRDiff with Stable Diffusion

This code is for FRdiff with Stable Diffusion (SDXL).

## Requirements
```
pip install torch transformers accelerate
pip install diffusers==0.26.3
```

## Sampling

If you want to sample SDXL, uncomment 34 line in sample.py.
```
python sample.py --num_steps 50 --interval 2 --prompt "a photo of an astronaut on a moon"
```
