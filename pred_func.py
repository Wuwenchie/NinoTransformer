from model import *  # 假設你的模型在 NinoGeoformer.py 中
import torch
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
from torch.utils.data import Dataset


class make_dataset_test(Dataset):
    def __init__(
        self,
        address,
        lon_range=(0, 1),
        lat_range=(0, 1),
        input_length=12,  # 增加一個變量來控制時間步數
    ):
        data_in = xr.open_dataset(address)
        self.lat = data_in["y"].values
        self.lon = data_in["x"].values
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.input_length = input_length  # 記錄輸入長度

        # 提取 SST 和 SSS
        sst = data_in["sst"].values
        sss = data_in["sss"].values
        sst = np.nan_to_num(sst)
        sss = np.nan_to_num(sss)
        sst[abs(sst) > 999] = 0
        sss[abs(sss) > 999] = 0

        # 將 SST 和 SSS 合併為輸入數據 (C=1)
        self.dataX = np.stack([sst, sss], axis=1)  # (time, 2, lat, lon)
        del sst, sss

    def getdatashape(self):
        return {"dataX.shape": self.dataX.shape}

    def selectregion(self):
        return {
            "lon: {}E to {}E".format(self.lon[self.lon_range[0]], self.lon[self.lon_range[1] - 1]),
            "lat: {}S to {}N".format(self.lat[self.lat_range[0]], self.lat[self.lat_range[1] - 1]),
        }

    def __len__(self):
        return self.dataX.shape[0] - self.input_length  # 確保有足夠的時間步

    def __getitem__(self, idx):
        return self.dataX[idx:idx + self.input_length]  # (T=12, C, H, W)


def func_pre(mypara, adr_model, adr_datain, adr_oridata):
    lead_max = mypara.output_length

    # 載入原始數據 (用於真值比對)
    data_ori = xr.open_dataset(adr_oridata)
    sst_ori = data_ori["sst"].values
    sss_ori = data_ori["sss"].values
    nino34 = data_ori["nino34"].values
    std_sst = data_ori["std_sst"].values
    # std_sst = np.nanmean(std_sst, axis=(1, 2))
    std_sss = data_ori["std_sss"].values  # 假設有 stdsal
    # std_sss = np.nanmean(std_sss, axis=(1, 2))

    # 合併 SST 和 SSS 真值
    var_ori_region = np.stack((sst_ori, sss_ori), axis=1)  # (T, C, H, W)
    stds = np.concatenate((std_sst[None], std_sss[None]), axis=0)
    del sst_ori_region, sss_ori_region, std_sst, std_sss

    # 測試數據集
    dataCS = make_dataset_test(
        address=adr_datain,
        lon_range=mypara.lon_range,
        lat_range=mypara.lat_range,
    )
    test_group = len(dataCS)
    print(dataCS.getdatashape())
    print(dataCS.selectregion())
    dataloader_test = DataLoader(dataCS, batch_size=mypara.batch_size_eval, shuffle=False)

    # 載入模型
    mymodel = NinoGeoformer(mypara).to(mypara.device)
    mymodel.load_state_dict(torch.load(adr_model))
    mymodel.eval()

    # 預測輸出維度
    n_channels = 2  # SST 和 SSS
    var_pred = np.zeros(
        [
            test_group,      # 測試樣本數
            lead_max,        # 預測的 lead time (時間步)
            n_channels,
            mypara.lat_range[1] - mypara.lat_range[0],
            mypara.lon_range[1] - mypara.lon_range[0],
        ]
    )

    # 模型預測
    ii = 0
    iii = 0
    with torch.no_grad():
        for input_var in dataloader_test:
            input_var = input_var.float().to(mypara.device).permute(0, 3, 1, 2)  # (B, C, H, W)
            input_var = input_var.unsqueeze(1)  # (B, T=1, C, H, W) 假設單一時間步輸入
            out_var = mymodel(input_var, predict_tar=None, train=False)  # (B, T, C, H, W)
            ii += out_var.shape[0]
            if torch.cuda.is_available():
                var_pred[iii:ii] = out_var.cpu().detach().numpy()
            else:
                var_pred[iii:ii] = out_var.detach().numpy()
            iii = ii
    del out_var, input_var, mymodel, dataCS, dataloader_test

    # 數據後處理
    len_data = test_group - lead_max
    print("len_data:", len_data)

    # 真值
    cut_var_true = var_ori_region[(12 + lead_max) - 1:] * stds[None, :, None, None]
    cut_nino_true = nino34[(12 + lead_max) - 1:]
    assert cut_nino_true.shape[0] == cut_var_true.shape[0] == len_data

    # 預測值
    cut_var_pred = np.zeros(
        [lead_max, len_data, var_pred.shape[2], var_pred.shape[3], var_pred.shape[4]]
    )
    cut_nino_pred = np.zeros([lead_max, len_data])
    for i in range(lead_max):
        l = i + 1
        cut_var_pred[i] = var_pred[lead_max - l:test_group - l, i] * stds[None, :, None, None]
        cut_nino_pred[i] = np.nanmean(
            cut_var_pred[
                i,
                :,
                mypara.lat_nino_relative[0]:mypara.lat_nino_relative[1],
                mypara.lon_nino_relative[0]:mypara.lon_nino_relative[1],
            ],
            axis=1,
        )
    assert cut_var_pred.shape[1] == cut_var_true.shape[0]

    return cut_var_pred, cut_var_true, cut_nino_pred, cut_nino_true
