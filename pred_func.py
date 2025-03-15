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
        lev_range=(0, 1),
        lon_range=(0, 1),
        lat_range=(0, 1),
    ):
        data_in = xr.open_dataset(address)
        self.lat = data_in["y"].values
        self.lon = data_in["x"].values
        self.lon_range = lon_range
        self.lat_range = lat_range

        # 提取 SST 和 SSS
        sst = data_in["sst"][
            :,
            :,
            lat_range[0]:lat_range[1],
            lon_range[0]:lon_range[1],
        ].values
        sss = data_in["salinityNor"][  # 假設數據集中有 salinityNor
            :,
            :,
            lev_range[0]:lev_range[1],
            lat_range[0]:lat_range[1],
            lon_range[0]:lon_range[1],
        ].values
        sst = np.nan_to_num(sst)
        sss = np.nan_to_num(sss)
        sst[abs(sst) > 999] = 0
        sss[abs(sss) > 999] = 0

        # 將 SST 和 SSS 合併為輸入數據 (C=2)
        self.dataX = np.concatenate((sst[:, :, None], sss[:, :, None]), axis=2)  # (T, H, W, C)
        del sst, sss

    def getdatashape(self):
        return {"dataX.shape": self.dataX.shape}

    def selectregion(self):
        return {
            "lon: {}E to {}E".format(self.lon[self.lon_range[0]], self.lon[self.lon_range[1] - 1]),
            "lat: {}S to {}N".format(self.lat[self.lat_range[0]], self.lat[self.lat_range[1] - 1]),
            "lev: {}m to {}m".format(self.lev[self.lev_range[0]], self.lev[self.lev_range[1] - 1]),
        }

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, idx):
        return self.dataX[idx]


def func_pre(mypara, adr_model, adr_datain, adr_oridata):
    lead_max = mypara.output_length

    # 載入原始數據 (用於真值比對)
    data_ori = xr.open_dataset(adr_oridata)
    sst_ori_region = data_ori["temperatureNor"][
        :,
        mypara.lev_range[0]:mypara.lev_range[1],
        mypara.lat_range[0]:mypara.lat_range[1],
        mypara.lon_range[0]:mypara.lon_range[1],
    ].values
    sss_ori_region = data_ori["salinityNor"][
        :,
        mypara.lev_range[0]:mypara.lev_range[1],
        mypara.lat_range[0]:mypara.lat_range[1],
        mypara.lon_range[0]:mypara.lon_range[1],
    ].values
    nino34 = data_ori["nino34"].values
    std_sst = data_ori["stdtemp"][mypara.lev_range[0]:mypara.lev_range[1]].values
    std_sst = np.nanmean(std_sst, axis=(1, 2))
    std_sss = data_ori["stdsal"][mypara.lev_range[0]:mypara.lev_range[1]].values  # 假設有 stdsal
    std_sss = np.nanmean(std_sss, axis=(1, 2))

    # 合併 SST 和 SSS 真值
    var_ori_region = np.concatenate((sst_ori_region[:, None], sss_ori_region[:, None]), axis=1)  # (T, C, H, W)
    stds = np.concatenate((std_sst[None], std_sss[None]), axis=0)
    del sst_ori_region, sss_ori_region, std_sst, std_sss

    # 測試數據集
    dataCS = make_dataset_test(
        address=adr_datain,
        lev_range=mypara.lev_range,
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
    n_lev = 2  # SST 和 SSS
    sst_lev = 0  # SST 在通道 0
    var_pred = np.zeros(
        [
            test_group,
            lead_max,
            n_lev,
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
                sst_lev,
                mypara.lat_nino_relative[0]:mypara.lat_nino_relative[1],
                mypara.lon_nino_relative[0]:mypara.lon_nino_relative[1],
            ],
            axis=(1, 2),
        )
    assert cut_var_pred.shape[1] == cut_var_true.shape[0]

    return cut_var_pred, cut_var_true, cut_nino_pred, cut_nino_true
