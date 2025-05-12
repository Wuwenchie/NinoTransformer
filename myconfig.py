import torch


class Mypara:
    def __init__(self):
        pass


mypara = Mypara()
mypara.device = torch.device("cuda:1")
# mypara.cpu_device = torch.device("cpu")  # CPU 設備
mypara.batch_size_train = 8
mypara.batch_size_eval = 10
mypara.num_epochs = 40
mypara.TFnum_epochs = 20
mypara.TFlr = 1.5e-5
mypara.early_stopping = True
mypara.patience = 4
mypara.warmup = 2000

# data related
mypara.adr_pretr = (
    "./data/training_data/GFDL-CM4_sss_sst_1850_1976.nc"
)
mypara.interval = 4
mypara.TraindataProportion = 0.9
mypara.all_group = 13000
mypara.adr_eval = (
    "./data/training_data/GFDL-CM4_sss_sst_1977_2009.nc"
)
mypara.needtauxy = True
mypara.need_sss_tauxy = True
mypara.input_channal = 7  # n_lev of 3D temperature
mypara.output_channal = 7
mypara.input_length = 12
mypara.output_length = 20
mypara.lev_range = (1, 8)
mypara.lon_range = (45, 165)
mypara.lat_range = (0, 51)
# nino34 region
mypara.lon_nino_relative = (49, 75)
mypara.lat_nino_relative = (15, 36)

# patch size
mypara.patch_size = (3, 4)
# mypara.H0 = int((mypara.lat_range[1] - mypara.lat_range[0]) / mypara.patch_size[0])
# mypara.W0 = int((mypara.lon_range[1] - mypara.lon_range[0]) / mypara.patch_size[1])
mypara.H0 = (23 - mypara.patch_size[0]) // mypara.patch_size[0] + 1  # 7
mypara.W0 = (72 - mypara.patch_size[1]) // mypara.patch_size[1] + 1  # 17
mypara.emb_spatial_size = mypara.H0 * mypara.W0

# model
mypara.model_savepath = "./model/"
mypara.seeds = 1
mypara.d_size = 128
mypara.nheads = 4
mypara.dim_feedforward = 512
mypara.dropout = 0.2
mypara.num_encoder_layers = 2
mypara.num_decoder_layers = 2


"""
import torch


class Mypara:
    def __init__(self):
        pass


mypara = Mypara()
mypara.device = torch.device("cuda:0")
mypara.batch_size_train = 8
mypara.batch_size_eval = 10
mypara.num_epochs = 40
mypara.TFnum_epochs = 20
mypara.TFlr = 1.5e-5
mypara.early_stopping = True
mypara.patience = 4
mypara.warmup = 2000

# data related
mypara.adr_pretr = (
    "./data/GFDL-CM4_sss_sst_1850_2009.nc"
)
mypara.interval = 4
mypara.TraindataProportion = 0.9
mypara.all_group = 13000
mypara.adr_eval = (
    "./data/GFDL-CM4_sss_sst_2010_2024.nc"
)
mypara.needtauxy = True
mypara.input_channal = 7  # n_lev of 3D temperature
mypara.output_channal = 7
mypara.input_length = 12
mypara.output_length = 24
mypara.lon_range = (-299, 59)
mypara.lat_range = (-55, 59)

# nino34 region
# mypara.lon_nino_relative = (49, 75)
# mypara.lat_nino_relative = (15, 36)
mypara.lon_nino_relative = (130, 180)
mypara.lat_nino_relative = (51, 61)

# patch size
mypara.patch_size = (3, 4)
# mypara.H0 = int((mypara.lat_range[1] - mypara.lat_range[0]) / mypara.patch_size[0])
# mypara.W0 = int((mypara.lon_range[1] - mypara.lon_range[0]) / mypara.patch_size[1])
mypara.H0 = (115 - mypara.patch_size[0]) // mypara.patch_size[0] + 1  # 7
mypara.W0 = (359 - mypara.patch_size[1]) // mypara.patch_size[1] + 1  # 17
mypara.emb_spatial_size = mypara.H0 * mypara.W0

# model
mypara.model_savepath = "./model/"
mypara.seeds = 1
mypara.d_size = 256
mypara.nheads = 4
mypara.dim_feedforward = 512
mypara.dropout = 0.2
mypara.num_encoder_layers = 4
mypara.num_decoder_layers = 4
"""
