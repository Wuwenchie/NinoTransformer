import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import detrend


def convert(file_path):
    # 讀取數據檔案
    # file_path = "./data/training_data/SSS/sos_1850_2009.nc"
    ds = xr.open_dataset(file_path)
    dataset = nc.Dataset(file_path)
    print(dataset.variables.keys())
    print(ds)
    
def sperate_with_time(file_path):
    file_path = "./data/testing_data/SST/tos_Omon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-203412.nc"
    ds = xr.open_dataset(file_path)
    print(ds)
    print(ds["time"])
    print(ds['tos'])
    ds_subset = ds.sel(time=slice("2015-01-16", "2024-12-16"))
    ds_subset.to_netcdf("sst_2015_2024.nc")

def nc_combine(file_paths):
    # file_paths = "C:/Users/miawu/nino/data/testing_data/SST/"       # 存放目標的檔案路徑
    #定義一個空列表來儲存該文件里的.nc數據路徑
    dir_path = [] 

    for file_name in os.listdir(file_paths):
        dir_path.append(file_paths+file_name)
    dir_path ##输出文件路徑

    print("success enter file...")

    data_new = [] ##建立一个空列表，存储逐日降雨数据
    for i in range(len(dir_path)):
        data = xr.open_dataset(dir_path[i])['tos']
        data_new.append(data) ##儲存数据
    da = xr.concat(data_new,dim='time') ##将數據以時間维度来進行拼接
    #print(da)
    print("process...")

    path_new = "C:/Users/miawu/nino/data/testing_data/" ##设置新路径
    #print(da.variable)
    da.to_netcdf(path_new + 'sss_2010_2024.nc') ##对拼接好的文件进行存储
    da.close() ##关闭文件
    print("combine success!")


def interpolated(file_path):
    ds = xr.open_dataset(file_path)  # 假設數據為 NetCDF 格式
    lat = ds['lat'].values  # 原始緯度
    lon = ds['lon'].values  # 原始經度
    sst = ds['tos'].values  # 目標數據

    print("success loading data...")

    # 目標插值網格（5° x 5°）
    new_lat = np.arange(-55, 60, 5)  # 目標緯度範圍 55°S ~ 60°N，每 5°
    new_lon = np.arange(0, 360, 5)   # 目標經度範圍 0°E ~ 355°E，每 5°

    # 建立新的經緯度網格
    new_lon_grid, new_lat_grid = np.meshgrid(new_lon, new_lat)

    # 進行插值（線性插值法）
    # new_sst = griddata((lon.ravel(), lat.ravel()), sss.ravel(), (new_lon_grid, new_lat_grid), method='linear')


    # 將 lon 與 lat 轉為 (N, 2) 格式
    points = np.column_stack((lon.ravel(), lat.ravel()))
    M, N = new_lon_grid.shape
    print("new grid shape:", M, N)

    # 初始化一個存儲插值結果的陣列，其 shape 為 (1920, M, N)
    new_sst = np.empty((sst.shape[0], M, N))

    for t in range(sst.shape[0]):
        # 取出 t 時間步的 sss 數據
        sst_t = sst[t, :, :]
        # 如果 sss_t 有 NaN 值，可選擇移除：
        mask = ~np.isnan(sst_t)
        points_valid = np.column_stack((lon.ravel()[mask.ravel()], lat.ravel()[mask.ravel()]))
        values_valid = sst_t.ravel()[mask.ravel()]
        
        # 如果沒有 NaN 值，也可直接：
        # values = sss_t.ravel()
        
        # 進行插值
        new_sst[t, :, :] = griddata(points_valid, values_valid, (new_lon_grid, new_lat_grid), method='linear')
        
        print(f"step {t} -- interpolated successful")


    # 轉換回 xarray 格式
    # new_ds = xr.Dataset({'sss': (['lat', 'lon', 'time'], new_sss)}, coords={'lat': new_lat, 'lon': new_lon})


    new_lat = new_lat_grid[:, 0]
    new_lon = new_lon_grid[0, :]

    # 創建 DataArray
    new_ds = xr.DataArray(new_sst,
                    dims=['time', 'lat', 'lon'],
                    coords={'time': ds['time'], 'lat': new_lat, 'lon': new_lon},
                    name='sst')


    # 存儲新的插值數據
    new_ds.to_netcdf("sst_interpolated_2010_2024.nc")
    print("successful save sst_interpolated data")


def anomaly(file_path):
    # 讀取 NetCDF SST 數據
    ds = xr.open_dataset(file_path)
    print(ds)
    data = ds["sss"]  # 假設 SST 形狀為 (time, lat, lon)

    print("successful loading data..")

    # 計算月氣候平均值 (1991-2020)
    climatology = data.sel(time=slice("2010-01", "2024-12")).groupby("time.month").mean(dim="time")
    print("successful loading data...")

    # 去除趨勢
    # print("start detrend...")
    # data_detrended = xr.apply_ufunc(detrend, data, axis=0)
    # print("success")

    # 計算距平
    print("start calculate anomalies...")
    anomalies = data.groupby("time.month") - climatology
    print("sucess")

    # 保存結果
    anomalies.to_netcdf("sss_anomalies.nc")
    print("successful save sss_anomalies.nc")
    


def norm(file_path):
    # normalization data
    # 讀取 NetCDF 檔案
    ds = xr.open_dataset(file_path)

    # 提取 SSS 數據
    sst_data = ds["sss"].values  # 形狀: (time, lat, lon)
    print("success loading data")

    # 計算均值和標準差（沿時間維度）
    mean = np.mean(sst_data, axis=0)
    std = np.std(sst_data, axis=0)

    # 避免除以 0
    std[std == 0] = 1e-6

    print("calculating...")
    # Z-score 標準化
    sst_normalized = (sst_data - mean) / std

    new_ds = xr.DataArray(sst_normalized,
                    dims=['time', 'lat', 'lon'],
                    coords={'time': ds['time'], 'lat': ds['lat'], 'lon': ds['lon']},
                    name='sss')

    path = "./data/"
    new_ds.to_netcdf(path + "sss_norm_2010_2024.nc")
    print("success to save data")

file_path = "./data/sss_anomalies_2010_2024.nc"
norm(file_path)

def merge(file_path_sss, file_path_sst):
    path = "./data/"
    # file_path_sss = path + "sss_norm_1850_2009.nc"
    # file_path_sst = path + "sst_norm_1850_2009.nc"
    ds_sss = xr.open_dataset(file_path_sss)
    ds_sst = xr.open_dataset(file_path_sst)
    ds_new = xr.merge([ds_sss, ds_sst])
    ds_new.to_netcdf(path + "GFDL-CM4_sss_sst_2010_2024.nc")

file_path_sss = "./data/sss_norm_2010_2024.nc"
file_path_sst = "./data/sst_norm_2010_2024.nc"
merge(file_path_sss, file_path_sst)
