# Stable Diffusion Sampling Method Comparison
This repository contains the code for my bachelor thesis "A Comparative Analysis of Stable Diffusion Sampling Methods".

Sampling methods that are compared:
* DDPM
* DDIM
* UniPC
* DPM-Solver

Evaluation metrics are:
* Fréchet Inception Distance for image quality
* Wall-clock time for efficiency

Installation Prerequisites:
* Python 3.10
* PyTorch
* Hugging Face Diffusers (for UniPC and DPM-Solver)
* NumPy

# Results on a NVIDIA GeForce RTX 3060 Ti
| **Method**  | **Sampling Steps** | **FID**  | **Time Taken (s)** | **Sample Size** |
|-------------|--------------------|----------|--------------------|-----------------|
| DDIM        | 20                 | 21.16    | 514.56             | 10000           |
| DDIM        | 50                 | 16.49    | 966.03             | 10000           |
| DDIM        | 80                 | **14.77**| 1662.62            | 10000           |
| DDIM        | 100                | 16.05    | 2053.78            | 10000           |
| DDIM        | 200                | 15.34    | 3846.22            | 10000           |
| DDPM        | 100                | 305.74   | 1915.24            | 10000           |
| DDPM        | 500                | 47.78    | 10422.12           | 10000           |
| DDPM        | 1000               | **18.18**| 20141.25           | 10000           |
| DPMSolver   | 5                  | 59.12    | 203.31             | 10000           |
| DPMSolver   | 10                 | 29.39    | 199.77             | 10000           |
| DPMSolver   | 15                 | 22.81    | 396.40             | 10000           |
| DPMSolver   | 20                 | 21.42    | 518.80             | 10000           |
| DPMSolver   | 30                 | 19.68    | 719.87             | 10000           |
| DPMSolver   | 50                 | 18.95    | 969.15             | 10000           |
| DPMSolver   | 75                 | 17.90    | 1453.81            | 10000           |
| DPMSolver   | 100                | **17.81**| 1931.91            | 10000           |
| UniPC       | 5                  | 54.27    | 107.94             | 10000           |
| UniPC       | 10                 | 27.49    | 211.59             | 10000           |
| UniPC       | 15                 | 22.97    | 619.70             | 10000           |
| UniPC       | 20                 | 21.34    | 831.82             | 10000           |
| UniPC       | 30                 | 19.27    | 1200.83            | 10000           |
| UniPC       | 50                 | 18.27    | 1573.21            | 10000           |
| UniPC       | 75                 | 18.31    | 1547.69            | 10000           |
| UniPC       | 100                | **17.37**| 2171.21            | 10000           |

# Acknowledgement

My code is based on 
* [pytorch_diffusion](https://github.com/pesser/pytorch_diffusion)
* [diffusion models pytorch](https://github.com/dome272/Diffusion-Models-pytorch/)
* [ddim](https://github.com/ermongroup/ddim)
* [diffusion DDIM pytorch](https://github.com/Alokia/diffusion-DDIM-pytorch)
* [UniPC](https://github.com/wl-zhao/UniPC)
And uses huggingface implementations (Apache License Version 2.0) for:
* [DPM-Solver](https://huggingface.co/docs/diffusers/en/api/schedulers/multistep_dpm_solver)
* [UniPC](https://huggingface.co/docs/diffusers/en/api/schedulers/unipc)

Furthermore these papers have been used as a basis for the algorithms:
```
@article{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={6840--6851},
  year={2020}
}
```
```
@article{song2020denoising,
  title={Denoising diffusion implicit models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2010.02502},
  year={2020}
}
```
```
@article{zhao2024unipc,
  title={Unipc: A unified predictor-corrector framework for fast sampling of diffusion models},
  author={Zhao, Wenliang and Bai, Lujia and Rao, Yongming and Zhou, Jie and Lu, Jiwen},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```
```
@article{lu2022dpm,
  title={Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps},
  author={Lu, Cheng and Zhou, Yuhao and Bao, Fan and Chen, Jianfei and Li, Chongxuan and Zhu, Jun},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={5775--5787},
  year={2022}
}
```
