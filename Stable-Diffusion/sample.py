import time, argparse

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

from frdiff import wrapping_frdiff

def image_grid(imgs, rows, cols, padding=0):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    padded_w = w + padding * 2
    padded_h = h + padding * 2

    grid_w = cols * padded_w
    grid_h = rows * padded_h
    grid = Image.new('RGBA', size=(grid_w, grid_h), color=(0, 0, 0, 0))

    for i, img in enumerate(imgs):
        x = i % cols
        y = i // cols
        x_pos = x * padded_w + padding
        y_pos = y * padded_h + padding
        grid.paste(img, (x_pos, y_pos))

    return grid

def main(opt):
    # Load Model
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", variant="fp16", torch_dtype=torch.float16)
    # pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(f'cuda:{opt.gpu}')

    # DDIM Sampling
    generator = torch.Generator().manual_seed(opt.seed)
    start = time.time()
    image1 = pipe(opt.prompt, generator=generator, num_inference_steps=opt.num_steps).images[0]
    time1 = time.time() - start
    
    # FRDiff Sampling
    wrapping_frdiff(pipe, opt.num_steps, opt.interval, bias=opt.bias, debug=False)
    
    generator = torch.Generator().manual_seed(opt.seed)
    start = time.time()
    image2 = pipe(opt.prompt, generator=generator, num_inference_steps=opt.num_steps).images[0]
    time2 = time.time() - start
    
    # Save grid
    image_grid([image1, image2], 1, 2).save('output.png')
    
    # Summary
    print("-" * 50)
    print(f'Sampling Time Summary \
            \n   - DDIM {opt.num_steps} Step : \t {time1: 0.3f}s \
            \n   - FRDiff {opt.num_steps} Step M={opt.interval}: {time2: 0.3f}s')
    print(f'FRDiff is {time1/time2: 0.2f}x faster than DDIM')
    print("-" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut on a moon")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--bias", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    opt = parser.parse_args()
    
    main(opt)