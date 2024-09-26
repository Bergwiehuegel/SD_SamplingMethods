import os
import torch
import time
from functions.ckpt_utils import download_cifar10_checkpoint
from functions.img_utils import save_image

from model.config import Config
from model.unet import Model

class DDPM:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda", n=10, run_id="test_run_ddpm"):
        self.noise_steps = noise_steps
        self.n = n
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.run_id = run_id

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    # linear noise schedule
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample(self, model, n, batch_size=10, save_dir="./generated_images"):
        num_batches = n // batch_size
        save_dir = os.path.join(save_dir, self.run_id)
        os.makedirs(save_dir, exist_ok=True)
        model.eval()

        start_time = time.time()
        # Algorithm 2: Sampling from the Diffusion Model adapted from https://arxiv.org/abs/2006.11239 / https://github.com/dome272/Diffusion-Models-pytorch / https://github.com/pesser/pytorch_diffusion
        with torch.no_grad():
            # generating in batches to not run out of memory
            for batch_idx in range(num_batches):
                # 1: xT ∼ N(0, I)
                x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(self.device)
                # 2: for t = T, ..., 1 do
                for i in reversed(range(1, self.noise_steps)):
                    t = (torch.ones(batch_size) * i).long().to(self.device)
                    predicted_noise = model(x, t)
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]
                    # 3:    z ∼ N(0, I) if t > 1, else z = 0
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    # 4: reconstruct previous noisy state
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                    # 5: end for

                # ensure pixel values are valid (clamping and normalize to [0,1])
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)

                # save images / x is the tensor containing the batch of images
                for img_idx, img in enumerate(x):
                    global_img_idx = batch_idx * batch_size + img_idx
                    save_path = os.path.join(save_dir, f"{self.run_id}_image_{global_img_idx}.png")
                    save_image(img, save_path)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Generated {n} images in {elapsed_time:.2f} seconds.")

        return save_dir, elapsed_time

if __name__ == '__main__':
    device = "cuda"
    config = Config()
    model = Model(config).to(device)
    ckpt = torch.load(download_cifar10_checkpoint(), map_location=device)
    model.load_state_dict(ckpt)
    diffusion = DDPM(img_size=32, device=device)
    output_dir, time_taken = diffusion.sample(model, n=10, save_dir="./generated_images")