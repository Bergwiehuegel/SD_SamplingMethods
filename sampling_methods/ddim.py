import os
import torch
import time
from functions.ckpt_utils import download_cifar10_checkpoint
from functions.img_utils import save_image

from model.config import Config
from model.unet import Model

import numpy as np

# BIG TODO maybe fix naming of variables to match paper, (like alpha_t instead of alpha - adapted from ddpm for now for my own better understanding)
class DDIM:
    def __init__(self, noise_steps=50, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda", n=10, run_id="test_run_ddim", training_steps=1000):
        self.noise_steps = noise_steps
        self.n = n
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.run_id = run_id
        self.training_steps = training_steps

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    # linear noise schedule
    def prepare_noise_schedule(self):
        return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.training_steps).to(self.device)

    def ddim_sample(self, model, n, batch_size=10, save_dir="./generated_images", eta=0.0):
        num_batches = n // batch_size
        save_dir = os.path.join(save_dir, self.run_id)
        os.makedirs(save_dir, exist_ok=True)
        model.eval() # TODO check my unet model for eval mode (used for only sampling and not training)

        a = self.training_steps // self.noise_steps
        time_steps = np.asarray(list(range(0, self.training_steps, a)))

        # fix indexing for alpha and alpha_hat
        time_steps = time_steps + 1
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        start_time = time.time()
        # sampling adapted from Denoising Diffusion Implicit Models: https://arxiv.org/abs/2010.02502 / https://github.com/ermongroup/ddim/ / the ddpm implementation / https://github.com/Alokia/diffusion-DDIM-pytorch/
        with torch.no_grad():
            for batch_idx in range(num_batches):
                # xT ∼ N(0, I)
                x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(self.device)

                for i in reversed(range(self.noise_steps)):
                    t = torch.full((x.shape[0],), time_steps[i], device=self.device, dtype=torch.long)
                    prev_t = torch.full((x.shape[0],), time_steps_prev[i], device=self.device, dtype=torch.long)

                    # get current and previous alpha_t values
                    alpha = self.alpha_hat[t].view(-1, 1, 1, 1)
                    alpha_prev = self.alpha_hat[prev_t].view(-1, 1, 1, 1)

                    # predicted noise
                    epsilon_theta_t = model(x, t)

                    # noise addition per step only relevant if eta > 0 (for ddim not the case but in here in case i want to test it slightly probabilistic)
                    sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
                    epsilon_t = torch.randn_like(x)

                    # EQ 12 from the paper (predicted x0 + direction pointing to xt + random noise (if eta > 0))
                    x = (
                        torch.sqrt(alpha_prev / alpha) * x +
                        (torch.sqrt(1 - alpha_prev - sigma_t ** 2) - torch.sqrt(
                            (alpha_prev * (1 - alpha)) / alpha)) * epsilon_theta_t +
                        sigma_t * epsilon_t
                    )

                # ensure pixel values are valid (clamping and normalize to [0,1])
                x = torch.clamp((x + 1) / 2, 0, 1)
                x = (x * 255).type(torch.uint8)

                # save images / x is the tensor containing the batch of images
                for img_idx, img in enumerate(x):
                    global_img_idx = batch_idx * batch_size + img_idx
                    save_path = os.path.join(save_dir, f"{self.run_id}_image_{global_img_idx}.png")
                    save_image(img, save_path)

        elapsed_time = time.time() - start_time
        print(f"Generated {n} images in {elapsed_time:.2f} seconds.")

        return save_dir, elapsed_time


if __name__ == '__main__':
    device = "cuda"
    config = Config()
    model = Model(config).to(device)
    ckpt = torch.load(download_cifar10_checkpoint(), map_location=device)
    model.load_state_dict(ckpt)
    diffusion = DDIM(img_size=32, device=device)

    output_dir, time_taken = diffusion.ddim_sample(model, n=10, save_dir="./generated_images")
