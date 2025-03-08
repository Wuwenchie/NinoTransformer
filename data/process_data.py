import xarray as xr
import os
import numpy as np
from scipy.interpolate import griddata
import netCDF4 as nc

def convert(file_path):
    # 讀取數據檔案
    # file_path = "./data/training_data/SSS/sos_1850_2009.nc"
    ds = xr.open_dataset(file_path)
    dataset = nc.Dataset(file_path)
    print(dataset.variables.keys())
    print(ds)


def sperate_with_time(file_path, sel_time_start, sel_time_finish, path_new, output_filename):
    ds = xr.open_dataset(file_path)
    print(ds)
    ds_subset = ds.sel(time=slice(sel_time_start, sel_time_finish))
    ds_subset.to_netcdf(path_new + output_filename)
    print(f"success sel time {sel_time_start} to {sel_time_finish}")


def nc_combine(file_paths, var, path_new, output_filename):
    # 定義一個空列表來儲存該文件里的.nc數據路徑
    dir_path = []

    for file_name in os.listdir(file_paths):
        dir_path.append(file_paths + file_name)

    dir_path  ##输出文件路徑
    print("success enter file...")

    data_new = []  ##建立一个空列表，存储逐日降雨数据
    for i in range(len(dir_path)):
        data = xr.open_dataset(dir_path[i])[var]
        data_new.append(data)  ##儲存数据
    da = xr.concat(data_new, dim='time')  ##将數據以時間维度来進行拼接
    print("process...")
    da.to_netcdf(path_new + output_filename)  ##对拼接好的文件进行存储
    da.close()  ##关闭文件
    print(f"combine success to {output_filename}")


def interpolated(file_path, var, out_var, path_new, output_filename):
    ds = xr.open_dataset(file_path)  # 讀取 NetCDF 數據
    lat = ds['lat'].values  # 原始緯度
    lon = ds['lon'].values  # 原始經度
    data = ds[var].values  # 目標數據

    print("success loading data...")

    # 目標插值網格（1° x 1°）
    new_lat = np.arange(-55, 61, 1)  # 目標緯度範圍 -55°S ~ 60°N，每 1°
    new_lon = np.arange(0, 360, 1)  # 目標經度範圍 0°E ~ 359°E，每 1°

    # 建立新的經緯度網格
    new_lon_grid, new_lat_grid = np.meshgrid(new_lon, new_lat)

    # 將原始經緯度轉為 (N, 2) 格式
    points = np.column_stack((lon.ravel(), lat.ravel()))
    M, N = new_lon_grid.shape
    print(f"New grid shape: {M} x {N}")

    # 初始化存儲插值結果的陣列，其 shape 為 (time, lat, lon)
    new_sst = np.empty((data.shape[0], M, N))

    for t in range(data.shape[0]):
        # 取出 t 時間步的 sst 數據
        data_t = data[t, :, :]

        # 移除 NaN 值以進行插值
        mask = ~np.isnan(data_t)
        points_valid = np.column_stack((lon.ravel()[mask.ravel()], lat.ravel()[mask.ravel()]))
        values_valid = data_t.ravel()[mask.ravel()]

        # 進行插值
        new_sst[t, :, :] = griddata(points_valid, values_valid, (new_lon_grid, new_lat_grid), method='linear')

        print(f"Step {t} -- interpolated successful")

    # 轉換回 xarray 格式
    new_ds = xr.DataArray(new_sst,
                          dims=['time', 'lat', 'lon'],
                          coords={'time': ds['time'], 'lat': new_lat, 'lon': new_lon},
                          name=out_var)

    # 存儲新的插值數據
    new_ds.to_netcdf(path_new + output_filename)
    print(f"Successfully saved {output_filename}")


def anomaly(file_path, var, sel_time_start, sel_time_finish, path_new, output_filename):
    # 讀取 NetCDF SST 數據
    ds = xr.open_dataset(file_path)
    print(ds)
    data = ds[var]  # 假設 SST 形狀為 (time, lat, lon)

    print("successful loading data..")

    # 計算月氣候平均值 (1991-2020)
    climatology = data.sel(time=slice(sel_time_start, sel_time_finish)).groupby("time.month").mean(dim="time")
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
    anomalies.to_netcdf(path_new + output_filename)
    print(f"successful save {output_filename}")


def norm(file_path, var, path_new, output_filename):
    # normalization data
    # 讀取 NetCDF 檔案
    ds = xr.open_dataset(file_path)

    # 提取 SSS 數據
    data = ds[var].values  # 形狀: (time, lat, lon)
    print("success loading data")

    # 計算均值和標準差（沿時間維度）
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # 避免除以 0
    std[std == 0] = 1e-6

    print("calculating...")
    # Z-score 標準化
    sst_normalized = (data - mean) / std

    sst_normalized.to_netcdf(path_new + output_filename)
    print(f"success save data to {output_filename}")


def merge(file_path_sss, file_path_sst, path_new, output_filename):
    # file_path_sss = path + "sss_norm_1850_2009.nc"
    # file_path_sst = path + "sst_norm_1850_2009.nc"
    ds_sss = xr.open_dataset(file_path_sss)
    ds_sst = xr.open_dataset(file_path_sst)
    ds_new = xr.merge([ds_sss, ds_sst])
    ds_new.to_netcdf(path_new + output_filename)
    print(f"success merge data to {output_filename}")


def training_data_pipeline():
    path = "./data/training_data/"

    # 2. 合併 SSS/SST 檔案
    sss_dir = path + "SSS/"
    sst_dir = path + "SST/"
    nc_combine(sss_dir, 'sos', sss_dir, "sss_1850_2009.nc")
    nc_combine(sst_dir, 'tos', sst_dir, "sst_1850_2009.nc")

    # 3. 插值到統一網格
    sss_file = sss_dir + "sss_1850_2009.nc"
    sst_file = sst_dir + "sst_1850_2009.nc"
    interpolated(sss_file, 'sos', 'sss', sss_dir, 'sss_interpolated_1850_2009.nc')
    interpolated(sst_file, 'tos', 'sst', sst_dir, 'sst_interpolated_1850_2009.nc')

    # 4. 計算異常值 (Anomaly)
    sss_interpolated_file = sss_dir + "sss_interpolated_1850_2009.nc"
    sst_interpolated_file = sst_dir + "sst_interpolated_1850_2009.nc"
    anomaly(sss_interpolated_file, 'sss', '1850-01', '2009-12', sss_dir, 'sss_anomaly_1850_2009.nc')
    anomaly(sst_interpolated_file, 'sst', '1850-01', '2009-12', sst_dir, 'sst_anomaly_1850_2009.nc')

    # 5. 標準化 (Normalization)
    sss_anomaly_file = sss_dir + "sss_anomalies_1850_2009.nc"
    sst_anomaly_file = sst_dir + "sst_anomalies_1850_2009.nc"
    norm(sss_anomaly_file, 'sss', sss_dir, 'sss_norm_1850_2009.nc')
    norm(sst_anomaly_file, 'sst', sst_dir, 'sst_norm_1850_2009.nc')

    # 6. 合併 SST 和 SSS 數據
    sss_norm_file = sss_dir + "sss_norm_1850_2009.nc"
    sst_norm_file = sst_dir + "sst_norm_1850_2009.nc"
    merge(sss_norm_file, sst_norm_file, path, 'GFDL-CM4_sss_sst_1850_2009.nc')

    print("Training data Pipeline execution completed!")


def testing_data_pipeline():
    path = "./data/testing_data/"
    sss_dir = path + "SSS/"
    sst_dir = path + "SST/"

    file_path_sst = path + "tos_Omon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-203412.nc"
    file_path_sss = path + "sos_Omon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-203412.nc"
    output_filename_sst = sst_dir + 'tos_Omon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-202412.nc'
    output_filename_sss = sss_dir + 'sos_Omon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-202412.nc'
    sperate_with_time(file_path_sst, '2015-01', '2014-12', sst_dir, output_filename_sst)
    sperate_with_time(file_path_sss, '2015-01', '2014-12', sss_dir, output_filename_sss)

    # 2. 合併 SSS/SST 檔案
    nc_combine(sss_dir, 'sos', sss_dir, "sss_2010_2024.nc")
    nc_combine(sst_dir, 'tos', sst_dir, "sst_2010_2024.nc")

    # 3. 插值到統一網格
    sss_file = sss_dir + "sss_2010_2024.nc"
    sst_file = sst_dir + "sst_2010_2024.nc"
    interpolated(sss_file, 'sos', 'sss', sss_dir, 'sss_interpolated_2010_2024.nc')
    interpolated(sst_file, 'tos', 'sst', sst_dir, 'sst_interpolated_2010_2024.nc')

    # 4. 計算異常值 (Anomaly)
    sss_interpolated_file = sss_dir + "sss_interpolated_2010_2024.nc"
    sst_interpolated_file = sst_dir + "sst_interpolated_2010_2024.nc"
    anomaly(sss_interpolated_file, 'sss', '2010-01', '2024-12', sss_dir, 'sss_anomaly_2010_2024.nc')
    anomaly(sst_interpolated_file, 'sst', '2010-01', '2024-12', sst_dir, 'sst_anomaly_2010_2024.nc')

    # 5. 標準化 (Normalization)
    sss_anomaly_file = sss_dir + "sss_anomalies_2010_2024.nc"
    sst_anomaly_file = sst_dir + "sst_anomalies_2010_2024.nc"
    norm(sss_anomaly_file, 'sss', sss_dir, 'sss_norm_2010_2024.nc')
    norm(sst_anomaly_file, 'sst', sst_dir, 'sst_norm_2010_2024.nc')

    # 6. 合併 SST 和 SSS 數據
    sss_norm_file = sss_dir + "sss_norm_2010_2024.nc"
    sst_norm_file = sst_dir + "sst_norm_2010_2024.nc"
    merge(sss_norm_file, sst_norm_file, path, 'GFDL-CM4_sss_sst_2010_2024.nc')

    print("Testing data Pipeline execution completed!")

if __name__ == "__main__":
    training_data_pipeline()
    testing_data_pipeline()
