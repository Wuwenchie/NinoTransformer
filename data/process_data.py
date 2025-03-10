import xarray as xr
import os
import numpy as np
from scipy.interpolate import griddata


def sperate_with_time(file_path, sel_time_start, sel_time_finish, path_new, output_filename):
    # file_path = "./data/testing_data/SST/tos_Omon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-203412.nc"
    ds = xr.open_dataset(file_path)
    print(ds)
    # print(ds["time"])
    # print(ds['tos'])
    ds_subset = ds.sel(time=slice(sel_time_start, sel_time_finish))
    ds_subset.to_netcdf(path_new + output_filename)
    print(f"success sel time {sel_time_start} to {sel_time_finish}")

# sperate_with_time()
def nc_combine(file_paths, var, path_new, output_filename):
    # file_paths = "C:/Users/miawu/nino/data/testing_data/SST/"       # 存放目標的檔案路徑
    #定義一個空列表來儲存該文件里的.nc數據路徑
    dir_path = []

    for file_name in os.listdir(file_paths):
        dir_path.append(file_paths+file_name)
        print(dir_path)

    # dir_path ##输出文件路徑

    # 檢查每個文件的時間範圍並排序
    time_ranges = []
    for path in dir_path:
        ds = xr.open_dataset(path)
        time_values = ds['time'].values
        time_ranges.append((path, time_values[0]))
        ds.close()
    dir_path.sort(key=lambda x: time_ranges[[p[0] for p in time_ranges].index(x)][1])
    print("Sorted file paths: ", dir_path)
   
    print("success enter file...")

    data_new = [] ##建立一个空列表，存储逐日降雨数据
    for i in range(len(dir_path)):
        data = xr.open_dataset(dir_path[i])[var]
        data_new.append(data) ##儲存数据
    da = xr.concat(data_new,dim='time') ##将數據以時間维度来進行拼接
    #print(da)
    print("process...")

    # path_new = "C:/Users/miawu/nino/data/testing_data/" ##设置新路径
    #print(da.variable)
    da.to_netcdf(path_new + output_filename) ##对拼接好的文件进行存储
    da.close() ##关闭文件
    print(f"combine success to {output_filename}")

def interpolated(file_path, var, path_new, output_filename):
    # 加載數據
    # ds, var_name = interpolated(file_path)  # 假設上面定義了檢查函數
    print("=======================")
    print("loading data...")
    ds = xr.open_dataset(file_path)
    print("Dataset structure:")
    print(ds)

    # 獲取 2D 坐標
    lat_2d = ds['lat'].values
    lon_2d = ds['lon'].values
    print("lat_2d shape:", lat_2d.shape)  # (1080, 1440)
    print("lon_2d shape:", lon_2d.shape)  # (1080, 1440)

    # 獲取原始維度坐標
    y_coords = ds['y'].values  # 形狀 (1080,)
    x_coords = ds['x'].values  # 形狀 (1440,)
    print("y_coords shape:", y_coords.shape)
    print("x_coords shape:", x_coords.shape)

    # 目標插值網格（5° x 5°）
    new_lat = np.arange(-55, 60, 1)  # 24 個點
    new_lon = np.arange(-299, 60, 1)   # 72 個點
    new_lat_2d, new_lon_2d = np.meshgrid(new_lat, new_lon)
    print("new_lat_2d shape:", new_lat_2d.shape)  # (24, 72)
    print("new_lon_2d shape:", new_lon_2d.shape)  # (24, 72)

    # 準備插值點
    # 創建原始網格的 (lon, lat) 點
    print("prepare interploted point")
    points = np.column_stack((lon_2d.ravel(), lat_2d.ravel()))  # 形狀 (1080*1440, 2)
    # 目標網格的 (lon, lat) 點
    xi = np.column_stack((new_lon_2d.ravel(), new_lat_2d.ravel()))  # 形狀 (24*72, 2)

    # 插值 lat_2d 和 lon_2d
    values_lat = lat_2d.ravel()  # 形狀 (1080*1440,)
    values_lon = lon_2d.ravel()  # 形狀 (1080*1440,)
    new_lat_2d_interp = griddata(points, values_lat, xi, method='linear').reshape(359, 115)
    new_lon_2d_interp = griddata(points, values_lon, xi, method='linear').reshape(359, 115)

    print("new_lat_2d_interp shape:", new_lat_2d_interp.shape)
    print("new_lon_2d_interp shape:", new_lon_2d_interp.shape)
    print("new_lat_2d_interp min:", new_lat_2d_interp.min(), "max:", new_lat_2d_interp.max())
    print("new_lon_2d_interp min:", new_lon_2d_interp.min(), "max:", new_lon_2d_interp.max())

    # 創建臨時坐標映射
    print("start interpolated...")
    ds_temp = ds.assign_coords(lat=('y', lat_2d.mean(axis=1)), lon=('x', lon_2d.mean(axis=0)))
    new_ds = ds_temp[var].interp(x=new_lon, y=new_lat, method='linear', kwargs={'fill_value': None})

    # 分配插值後的 2D 坐標
    new_ds = new_ds.assign_coords(lat=('y', new_lat_2d_interp[0, :]), lon=('x', new_lon_2d_interp[:, 0]))
    # 注意：這裡只分配 1D 近似，因為 assign_coords 期望每個維度對應一個 1D 陣列
    # 如果需要 2D 坐標，可以使用多維坐標，但需要額外處理

    # 檢查插值結果
    print("Interpolated shape:", new_ds.shape)
    print("Interpolated min:", new_ds.min().item(), "max:", new_ds.max().item())

    # 檢查 Nino 3.4 區域
    nino_ds = new_ds.sel(y=slice(-5, 5), x=slice(-170, -120))
    print("Nino 3.4 region min:", nino_ds.min().item(), "max:", nino_ds.max().item())

    # 處理 nan 值
    if np.any(np.isnan(new_ds)):
        new_ds = new_ds.fillna(new_ds.mean(dim=['y', 'x']))
        print("After filling NaN, Nino 3.4 region min:", nino_ds.min().item())

    # 儲存插值結果
    # new_ds.to_netcdf("sst_interpolated_1970_2009.nc")
   # 轉換回 xarray 格式
    # new_xr = xr.DataArray(new_ds,
                        #   dims=['time', 'lat', 'lon'],
                        #   coords={'time': ds['time'], 'lat': new_lat, 'lon': new_lon},
                        #   name=out_var)
    # new_ds = new_ds.rename({var: out_var})
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


def norm(file_path, var, path_new, output_filename, out_var):
    # normalization data
    # 讀取 NetCDF 檔案
    ds = xr.open_dataset(file_path)
    print(ds)
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

    new_ds = xr.DataArray(sst_normalized,
                    dims=['time', 'y', 'x'],
                    coords={'time': ds['time'], 'y': ds['y'], 'x': ds['x']},
                    name=out_var)

    new_ds.to_netcdf(path_new + output_filename)
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
    # nc_combine(sss_dir, 'sos', sss_dir, "sss_1850_2009.nc")
    # nc_combine(sst_dir, 'tos', sst_dir, "sst_1850_2009.nc")

    # 3. 插值到統一網格
    sss_file = sss_dir + "sss_1850_2009.nc"
    sst_file = sst_dir + "sst_1850_2009.nc"
    # ds = xr.open_dataset(sss_file)
    # print(ds)
    # interpolated(sss_file, 'sos', sss_dir, 'sss_interpolated_1850_2009.nc')
    # interpolated(sst_file, 'tos', sst_dir, 'sst_interpolated_1850_2009.nc')

    # 4. 計算異常值 (Anomaly)
    sss_interpolated_file = sss_dir + "sss_interpolated_1850_2009.nc"
    sst_interpolated_file = sst_dir + "sst_interpolated_1850_2009.nc"
    # anomaly(sss_interpolated_file, 'sos', '1850-01-16', '2009-12-16', sss_dir, 'sss_anomaly_1850_2009.nc')
    # anomaly(sst_interpolated_file, 'tos', '1850-01-16', '2009-12-16 12', sst_dir, 'sst_anomaly_1850_2009.nc')

    # 5. 標準化 (Normalization)
    sss_anomaly_file = sss_dir + "sss_anomaly_1850_2009.nc"
    sst_anomaly_file = sst_dir + "sst_anomaly_1850_2009.nc"
    norm(sss_anomaly_file, 'sos', sss_dir, 'sss_norm_1850_2009.nc', 'sss')
    norm(sst_anomaly_file, 'tos', sst_dir, 'sst_norm_1850_2009.nc', 'sst')

    # 6. 合併 SST 和 SSS 數據
    sss_norm_file = sss_dir + "sss_norm_1850_2009.nc"
    sst_norm_file = sst_dir + "sst_norm_1850_2009.nc"
    merge(sss_norm_file, sst_norm_file, path, 'GFDL-CM4_sss_sst_1850_2009.nc')

    print("Training data Pipeline execution completed!")

def testing_data_pipeline():
    path = "./data/testing_data/"
    # path = "/home/g492/Downloads/wenchieh/nino/data/testing_data/"
    sss_dir = path + "SSS/"
    sst_dir = path + "SST/"
    
    # file_path_sst = os.path.join(path, file_path_sst_1)
    file_path_sss = path + "sos_Omon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-203412.nc"
    file_path_sst = path + "tos_Omon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-203412.nc"
    output_filename_sss = sss_dir + 'sos_Omon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-202412.nc'
    output_filename_sst = sst_dir + 'tos_Omon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-202412.nc'
    sperate_with_time(file_path_sss, '2015-01-16', '2014-12-16', sss_dir, output_filename_sss)
    sperate_with_time(file_path_sst, '2015-01-16', '2014-12-16', sst_dir, output_filename_sst)
    

    # 2. 合併 SSS/SST 檔案
    nc_combine(sss_dir, 'sos', sss_dir, "sss_2010_2024.nc")
    nc_combine(sst_dir, 'tos', sst_dir, "sst_2010_2024.nc")

    # 3. 插值到統一網格
    sss_file = sss_dir + "sss_2010_2024.nc"
    sst_file = sst_dir + "sst_2010_2024.nc"
    interpolated(sss_file, 'sos', sss_dir, 'sss_interpolated_2010_2024.nc')
    interpolated(sst_file, 'tos', sst_dir, 'sst_interpolated_2010_2024.nc')

    # 4. 計算異常值 (Anomaly)
    sss_interpolated_file = sss_dir + "sss_interpolated_2010_2024.nc"
    sst_interpolated_file = sst_dir + "sst_interpolated_2010_2024.nc"
    anomaly(sss_interpolated_file, 'sos', '2010-01-16', '2024-12-16', sss_dir, 'sss_anomaly_2010_2024.nc')
    anomaly(sst_interpolated_file, 'tos', '2010-01-16', '2024-12-16', sst_dir, 'sst_anomaly_2010_2024.nc')

    # 5. 標準化 (Normalization)
    sss_anomaly_file = sss_dir + "sss_anomaly_2010_2024.nc"
    sst_anomaly_file = sst_dir + "sst_anomaly_2010_2024.nc"
    norm(sss_anomaly_file, 'sos', sss_dir, 'sss_norm_2010_2024.nc', 'sss')
    norm(sst_anomaly_file, 'tos', sst_dir, 'sst_norm_2010_2024.nc', 'sst')

    # 6. 合併 SST 和 SSS 數據
    sss_norm_file = sss_dir + "sss_norm_2010_2024.nc"
    sst_norm_file = sst_dir + "sst_norm_2010_2024.nc"
    merge(sss_norm_file, sst_norm_file, path, 'GFDL-CM4_sss_sst_2010_2024.nc')

    print("Testing data Pipeline execution completed!")

training_data_pipeline()
testing_data_pipeline()
