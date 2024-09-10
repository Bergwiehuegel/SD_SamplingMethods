import os
import torch
import csv
import uuid
from functions.ckpt_utils import download_cifar10_checkpoint
from functions.img_utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths
from model.config import Config
from model.unet import Model

# sampling methods
from sampling_methods.ddim import DDIM
from sampling_methods.ddpm import DDPM
from sampling_methods.unipc import UniPC
from sampling_methods.dpm_solver import DPMSolver

if __name__ == '__main__':
    device = "cuda"
    config = Config()
    model = Model(config).to(device)
    ckpt = torch.load(download_cifar10_checkpoint(), map_location=device)
    model.load_state_dict(ckpt)

    run_id = str(uuid.uuid4())

    ddim_steps = 50  # (DDIM) https://arxiv.org/abs/2010.02502 states 20 to 100 sampling steps are comparable to ddpm results
    ddpm_steps = 1000  # (DDPM) https://arxiv.org/abs/2006.11239 states T = 1000 as a baseline in sampling
    unipc_steps = 5 # (UniPC) https://arxiv.org/abs/2302.04867 states 5 to 10 steps for fast sampling
    dpm_steps = 50 # (DPM-Solver) https://arxiv.org/abs/2206.00927 states 10 to 20 steps as fast sampling and 50 for more accuracy

    sample_size = 10000
    # test_size = 10

    # generate and safe images for FID comparison
    ddim_diffusion = DDIM(noise_steps=ddim_steps, img_size=32, device=device, run_id=f"ddim{run_id}")
    ddim_output_dir, ddim_time_taken = ddim_diffusion.ddim_sample(model, n=sample_size, save_dir="./generated_images")

    ddpm_diffusion = DDPM(noise_steps=ddpm_steps, img_size=32, device=device, run_id=f"ddpm_{run_id}")
    ddpm_output_dir, ddpm_time_taken = ddpm_diffusion.sample(model, n=sample_size, save_dir="./generated_images")

    unipc_diffusion = UniPC(img_size=32, device=device, run_id=f"unipc_{run_id}")
    unipc_output_dir, unipc_time_taken = unipc_diffusion.sample(model, n=sample_size, num_timesteps=unipc_steps, save_dir="./generated_images")

    dpm_solver = DPMSolver(img_size=32, device=device, run_id=f"dpm_{run_id}")
    dpm_output_dir, dpm_time_taken = dpm_solver.sample(model, n=sample_size, num_timesteps=dpm_steps, save_dir="./generated_images")

    # FID calculation with https://github.com/mseitzer/pytorch-fid
    real_images_path = './cifar10_real_images'
    ddim_fid = calculate_fid_given_paths([real_images_path, ddim_output_dir], batch_size=50, device=device, dims=2048)
    ddpm_fid = calculate_fid_given_paths([real_images_path, ddpm_output_dir], batch_size=50, device=device, dims=2048)
    unipc_fid = calculate_fid_given_paths([real_images_path, unipc_output_dir], batch_size=50, device=device, dims=2048)
    dpm_fid = calculate_fid_given_paths([real_images_path, dpm_output_dir], batch_size=50, device=device, dims=2048)

    # TODO but save metrics in csv for now (testing and thinking about it later)
    csv_file = 'sampling_metrics.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Run_ID', 'Method', 'Time_Taken (s)', 'FID', 'Sampling_Steps', 'Sample_Size'])
        writer.writerow([run_id, 'DDIM', ddim_time_taken, ddim_fid, ddim_steps, sample_size])
        writer.writerow([run_id, 'DDPM', ddpm_time_taken, ddpm_fid, ddpm_steps, sample_size])
        writer.writerow([run_id, 'UniPC', unipc_time_taken, unipc_fid, unipc_steps, sample_size])
        writer.writerow([run_id, 'DPMSolver', dpm_time_taken, dpm_fid, dpm_steps, sample_size])

    print(f"Metrics saved to {csv_file}")