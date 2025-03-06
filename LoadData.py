import numpy as np
from torch.utils.data import Dataset, IterableDataset
import xarray as xr
import random
from myconfig import *

class NinoDataset(IterableDataset):
    def __init__(self, mypara):
        self.mypara = mypara
        data_in = xr.open_dataset(mypara.adr_pretr)
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.input_length = mypara.input_length
        self.output_length = mypara.output_length
        self.all_group = mypara.all_group

        # 讀取 SST 和 SSS
        sst = data_in["sst"].values
        sss = data_in["sss"].values
        sst = np.nan_to_num(sst)
        sss = np.nan_to_num(sss)
        sst[abs(sst) > 999] = 0
        sss[abs(sss) > 999] = 0

        # 合併 SST 和 SSS
        self.field_data = np.stack([sst, sss], axis=1)  # (time, 2, lat, lon)
        print("field_data shape:", self.field_data.shape)

    def __iter__(self):
        st_min = self.input_length - 1
        ed_max = self.field_data.shape[0] - self.output_length
        for _ in range(self.all_group):
            rd_m = random.randint(0, self.field_data.shape[0] - 1)
            rd = random.randint(st_min, ed_max - 1)
            dataX = self.field_data[rd - self.input_length + 1:rd + 1]
            dataY = self.field_data[rd + 1:rd + self.output_length + 1]
            yield dataX, dataY

class NinoTestDataset(Dataset):
    def __init__(self, mypara, ngroup):
        self.mypara = mypara
        data_in = xr.open_dataset(mypara.adr_eval)
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lat_range = mypara.lat_range
        self.lon_range = mypara.lon_range

        # 輸入數據
        sst_in = data_in["sst"].values
        sss_in = data_in["sss"].values
        sst_in = np.nan_to_num(sst_in)
        sss_in = np.nan_to_num(sss_in)
        sst_in[abs(sst_in) > 999] = 0
        sss_in[abs(sss_in) > 999] = 0
        field_data_in = np.stack([sst_in, sss_in], axis=1)

        # 輸出數據
        sst_out = data_in["sst"].values
        sss_out = data_in["sss"].values
        sst_out = np.nan_to_num(sst_out)
        sss_out = np.nan_to_num(sss_out)
        sst_out[abs(sst_out) > 999] = 0
        sss_out[abs(sss_out) > 999] = 0
        field_data_out = np.stack([sst_out, sss_out], axis=1)

        self.dataX, self.dataY = self.deal_testdata(field_data_in, field_data_out, ngroup)

    def deal_testdata(self, field_data_in, field_data_out, ngroup):     # 從固定範圍內隨機選擇 ngroup 個樣本
        out_field_x = np.zeros([ngroup, self.mypara.input_length, 2, field_data_in.shape[2], field_data_in.shape[3]])
        out_field_y = np.zeros([ngroup, self.mypara.output_length, 2, field_data_out.shape[2], field_data_out.shape[3]])
        for j in range(ngroup):
            rd = random.randint(0, field_data_in.shape[0] - 1)
            # out_field_x[j] = field_data_in[rd]
            # out_field_y[j] = field_data_out[rd]
            out_field_x[j] = field_data_in[rd:rd + self.mypara.input_length]
            out_field_y[j] = field_data_out[rd:rd + self.mypara.output_length]
        return out_field_x, out_field_y

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, idx):
        return self.dataX[idx], self.dataY[idx]
    

# mydata = NinoDataset(mypara)
