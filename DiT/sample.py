# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
from torch import nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import torch.utils.checkpoint as checkpoint
class BUFFER :
    t = None
    cnt = -1


b = BUFFER()



def del_buffer(model) :
    #for n,m in model.module.named_modules() :
    for n,m in model.named_modules() :
        if hasattr(m, 'GLOBAL_BUFFER') :
            delattr(m, 'GLOBAL_BUFFER')

def add_buffer(model, clear_memory=True) :
    b = BUFFER()
    #for n,m in model.module.named_modules() :
    for n,m in model.named_modules() :
        m.GLOBAL_BUFFER = b
        m.name = n

    #for n,m in model.module.named_modules() :
    for n,m in model.named_modules() :
        if hasattr(m , 'memory') :
            m.memory = None


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    #torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict = False)

    model = model.float()
    #model.eval()  # important!

    #model = torch.nn.DataParallel(model)


    diffusion = create_diffusion(str(args.steps))
    diffusion_ref = create_diffusion("50")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    #class_labels = [360]

    # Create sampling noise:
    #import pdb; pdb.set_trace()
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    z1 = z
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)


    # Sample images:

    model.train() #??

    optim_params = []

    gate1 = nn.Parameter(torch.zeros(args.steps-1)+0.1)
    for n,m in model.named_modules() :
        m.gate1 = gate1

    optim_params = [gate1]
    for n,m in model.named_parameters() :
        if n.endswith('gate1') :
            m.requires_grad = True
        else :
            m.requires_grad = False
    optim = torch.optim.Adam( optim_params, lr=args.lr )
    import wandb
    prefix = f'{args.steps}_{args.lr}_{args.wgt}'
    wandb.init(project="DiT-Gate-Optim", name = prefix)
    import random
    for i in range(100) :
        #class_labels = [100,200,300,400]
        #class_labels = [random.randint(0,1000-1), random.randint(0,1000-1)]
        class_labels = [random.randint(0,1000-1)]

        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        #create reference

        del_buffer(model)

        with torch.no_grad() :
            samples_ref = diffusion_ref.ddim_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
        add_buffer(model)
        samples = diffusion.ddim_sample_loop(
            model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )

        loss = (samples_ref - samples).pow(2).mean()

        gate_loss = 0.

        gate1 = model.gate1
        print( torch.sigmoid(gate1).round() )

        gate_loss += nn.functional.relu(torch.sigmoid(gate1)*2 - 1).sum()

        wandb.log({
            'mse' : loss.item(), 
            'gate' : gate_loss.item()
            })

        loss += args.wgt*gate_loss
        print("LOSS : ", loss.item() ,gate_loss.item())

        loss.backward()
        print('========BACKWARD END=========')
        optim.step()
        optim.zero_grad()


    for n,m in model.named_modules() :
        if hasattr(m,'gate1') :
            print(n , torch.round(torch.sigmoid(m.gate1)))
            print("")


    del_buffer(model)
    model.eval()  # important!
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    z = z1
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)


    import time
    if args.sampler == 'ddim' :
        with torch.no_grad() :
            print("GO FR BRR")
            add_buffer(model)
            t1 = time.time()
            samples = diffusion.ddim_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            t2 = time.time()
            del_buffer(model)
            print("GO REF BRR")
            t3 = time.time()
            samples2 = diffusion_ref.ddim_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            t4 = time.time()
    elif args.sampler == 'ddpm' :
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    samples2, _ = samples2.chunk(2, dim=0)  # Remove null class samples
    samples2 = vae.decode(samples2 / 0.18215).sample

    # Save and display images:
    save_image(samples, f'{prefix}_sample.png', nrow=4, normalize=True, value_range=(-1, 1))
    save_image(samples2, "sample2.png", nrow=4, normalize=True, value_range=(-1, 1))

    with torch.no_grad() :
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        z = torch.cat([z, z], 0)
        print("GO FR BRR")
        add_buffer(model)
        t1 = time.time()
        samples = diffusion.ddim_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        t2 = time.time()
        del_buffer(model)
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample
        save_image(samples, f'{prefix}_sample2.png', nrow=4, normalize=True, value_range=(-1, 1))
    
    with torch.no_grad() :
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        z = torch.cat([z, z], 0)
        print("GO FR BRR")
        add_buffer(model)
        t1 = time.time()
        samples = diffusion.ddim_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        t2 = time.time()
        del_buffer(model)
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample
        save_image(samples, f'{prefix}_sample3.png', nrow=4, normalize=True, value_range=(-1, 1))
    
    print(f'FR  : {t2-t1}')
    print(f'REF : {t4-t3}')
    
    import pdb; pdb.set_trace()
    torch.save(model.state_dict(), f'{prefix}_model.ckpt')
    """
    #plot diffrences
    import matplotlib.pyplot as plt
    for n, m in model.named_modules():
        if hasattr(m, 'mem1_inp') :
            print(n)
            #print([ "{:.5f}".format(v) for v in m.mem1_inp_diff ])
            #print([ "{:.5f}".format(v) for v in m.mem1_oup_diff ])
            #print([ "{:.5f}".format(v) for v in m.mem2_inp_diff ])
            #print([ "{:.5f}".format(v) for v in m.mem2_oup_diff ])
            plt.clf()
            plt.plot(m.mem1_inp_diff)
            plt.plot(m.mem1_oup_diff)
            plt.plot(m.mem2_inp_diff)
            plt.plot(m.mem2_oup_diff)
            plt.plot(m.mem3)
            plt.legend(["inp1","oup1","inp2","oup2", "mem3"])
            #plt.ylim([0,1])
            plt.savefig(f'layer/{n}.png')
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampler", type=str, choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--wgt", type=float, default=3e-4)
    args = parser.parse_args()
    main(args)
