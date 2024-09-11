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
