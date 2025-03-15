import numpy as np
import xarray as xr

# 假設數據集路徑
data_path = "./data/GODAS_up150m_temp_sss_nino_kb.nc"
data = xr.open_dataset(data_path)

# 提取原始 SST 和 SSS
sst_raw = data["sst"].values  # 假設物理量字段名為 "temperature"
sss_raw = data["sss"].values     # 假設物理量字段名為 "salinity"

# 選擇範圍 (例如表面層和整個空間)
sst_raw = sst_raw[:, :, :]  # (T, lat, lon)
sss_raw = sss_raw[:, :, :]  # (T, lat, lon)

# 可選：限制到特定區域 (例如 Nino 3.4)
lat_nino = slice(mypara.lat_nino_relative[0], mypara.lat_nino_relative[1])
lon_nino = slice(mypara.lon_nino_relative[0], mypara.lon_nino_relative[1])
sst_raw_nino = sst_raw[:, lat_nino, lon_nino]
sss_raw_nino = sss_raw[:, lat_nino, lon_nino]

# 方法 2：對每個空間點計算標準差，然後平均
std_sst_spatial = np.nanstd(sst_raw, axis=0)  # (lat, lon)
std_sss_spatial = np.nanstd(sss_raw, axis=0)  # (lat, lon)
std_sst = np.nanmean(std_sst_spatial)         # 空間平均
std_sss = np.nanmean(std_sss_spatial)

print(f"std_sst: {std_sst}, std_sss: {std_sss}")
