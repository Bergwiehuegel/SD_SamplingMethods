#adapted from https://github.com/ermongroup/ddim
class Config:
    class ModelConfig:
        def __init__(self):
            self.ch = 128
            self.out_ch = 3
            self.ch_mult = (1, 2, 2, 2)
            self.num_res_blocks = 2
            self.attn_resolutions = (16,)
            self.dropout = 0.1
            self.in_channels = 3
            self.resamp_with_conv = True
            self.type = "default"

    class DataConfig:
        def __init__(self):
            self.image_size = 32

    class DiffusionConfig:
        def __init__(self):
            self.num_diffusion_timesteps = 1000

    def __init__(self):
        self.model = self.ModelConfig()
        self.data = self.DataConfig()
        self.diffusion = self.DiffusionConfig()