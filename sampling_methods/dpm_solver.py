import os
import torch
import time
from functions.ckpt_utils import download_cifar10_checkpoint
from functions.img_utils import save_image
from model.config import Config
from model.unet import Model
from diffusers import DPMSolverMultistepScheduler 

class DPMSolver:
    def __init__(self, num_train_timesteps=1000, img_size=32, device="cuda", n=10, run_id="test_run_dpm", solver_order=3): # solver order 3 for unconditional sampling
        self.num_train_timesteps = num_train_timesteps
        self.img_size = img_size
        self.device = device
        self.n = n
        self.run_id = run_id
        self.solver_order = solver_order
        self.initialize_scheduler()

    def initialize_scheduler(self):
        self.scheduler = DPMSolverMultistepScheduler.from_config({
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "num_train_timesteps": self.num_train_timesteps,
            "solver_order": self.solver_order,
        })

    def sample(self, model, n, batch_size=10, num_timesteps=50, save_dir="./generated_images"):
        num_batches = n // batch_size
        save_dir = os.path.join(save_dir, self.run_id)
        os.makedirs(save_dir, exist_ok=True)
        model.eval()

        start_time = time.time()

        self.scheduler.set_timesteps(num_timesteps)

        with torch.no_grad():
            for batch_idx in range(num_batches):

                self.initialize_scheduler()
                self.scheduler.set_timesteps(num_timesteps)

                # initial noise x ∼ N(0, I)
                x = torch.randn((batch_size, 3, self.img_size, self.img_size), device=self.device)

                # sampling loop
                for t in self.scheduler.timesteps:
                    timesteps = torch.full((batch_size,), t, device=x.device)

                    # 
                    predicted_noise = model(x, timesteps)
                    previous_noisy_sample = self.scheduler.step(predicted_noise, t, x).prev_sample
                    x = previous_noisy_sample

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
    sampler = DPMSolver(img_size=32, device=device)
    output_dir, time_taken = sampler.sample(model, n=20, save_dir="./generated_images")
