import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
# import netCDF4 as nc
import xarray as xr
from scipy.interpolate import griddata
import os
import pandas as pd


def plot_dataset(dataset: xr.Dataset):
    data = dataset.copy()
    projection = ccrs.Mercator()
    crs = ccrs.PlateCarree()
    plt.figure(figsize=(16, 9), dpi=150)
    ax = plt.axes(projection=projection, frameon=True)

    gl = ax.gridlines(crs=crs, draw_labels=True,
                      linewidth=.6, color='gray', alpha=0.5, linestyle='-.')
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    # To plot borders and coastlines, we can use cartopy feature
    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
    ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.3)

    # Now, we will specify extent of our map in minimum/maximum longitude/latitude
    # Note that these values are specified in degrees of longitude and degrees of latitude
    # However, we can specify them in any crs that we want, but we need to provide appropriate
    # crs argument in ax.set_extent
    # crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees
    lon_min = -299
    lon_max = 60
    lat_min = -55
    lat_max = 60

    ##### WE ADDED THESE LINES #####
    cbar_kwargs = {'orientation': 'horizontal', 'shrink': 0.6, "pad": .05, 'aspect': 40,
                   'label': 'sea surface temperature'}
    data['sssNor_out'].isel(group=0, Tout=0).plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs, levels=21, vmin=-1, vmax=1, cmap='RdBu_r')
    ################################

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)
    # plt.title(f"Temperature anomaly over Europe in {dataset.valid_time.dt.strftime('%B %Y').values}")
    plt.show()

def convert_lon(file_path, out_dir, output_filename):
    ds = xr.open_dataset(file_path)
    print(ds)
    # 轉換經度範圍到 0~360
    ds["x"] = (ds["x"] + 360) % 360

    # 排序經度（由小到大）
    ds = ds.sortby("x")
    print(ds)

    output_file = os.path.join(out_dir, output_filename)
    ds.to_netcdf(output_file)
    print(f"success save {output_filename}")

def interpolated(file_path, var, path_new, name):
    # 加載數據
    # ds, var_name = interpolated(file_path)  # 假設上面定義了檢查函數
    print("=======================")
    print("loading data...")
    ds = xr.open_dataset(file_path)
    print("Dataset structure:")
    print(ds)
    shape = ds[var].values.shape
    print(shape)
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

    # # 平移經度到 -180° 到 180°
    # shift = int((360 - 60.3) / 0.25)  # 計算需要平移的格點數
    # ds = ds.roll(x=-shift, roll_coords=True)
    # ds['x'] = (ds['x'] + 360) % 360  # 將經度轉換到 -180° 到 180°


    # 目標插值網格（5° x 5°）
    # 緯度區域設定：高解析赤道區
    lat_south = np.arange(-20, -5, 1.0)         # -20, -19, ..., -6
    lat_equator = np.arange(-5, 5.5, 0.5)       # -5.0, -4.5, ..., 5.0
    lat_north = np.arange(6, 21, 1.0)           # 6, 7, ..., 20

    # 合併為非均勻解析度的緯度陣列
    new_lat = np.concatenate([lat_south, lat_equator, lat_north])

    # new_lat = np.arange(-20, 21, 1.0)  # 24 個點
    new_lon = np.arange(2, 359, 2.0)   # 72 個點
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
    new_lat_2d_interp = griddata(points, values_lat, xi, method='linear').reshape(179, 51)
    new_lon_2d_interp = griddata(points, values_lon, xi, method='linear').reshape(179, 51)

    print("new_lat_2d_interp shape:", new_lat_2d_interp.shape)
    print("new_lon_2d_interp shape:", new_lon_2d_interp.shape)
    print("new_lat_2d_interp min:", new_lat_2d_interp.min(), "max:", new_lat_2d_interp.max())
    print("new_lon_2d_interp min:", new_lon_2d_interp.min(), "max:", new_lon_2d_interp.max())

    # 創建臨時坐標映射
    print("start interpolated...")
    ds_temp = ds.assign_coords(lat=('y', lat_2d.mean(axis=1)), lon=('x', lon_2d.mean(axis=0)))

    for t in range(shape[0]):
        new_ds = ds_temp[var].isel(time=t).interp(x=new_lon, y=new_lat, method='linear', kwargs={'fill_value': None})

        # 分配插值後的 2D 坐標
        new_ds = new_ds.assign_coords(lat=('y', new_lat_2d_interp[0, :]), lon=('x', new_lon_2d_interp[:, 0]))
        # 注意：這裡只分配 1D 近似，因為 assign_coords 期望每個維度對應一個 1D 陣列
        # 如果需要 2D 坐標，可以使用多維坐標，但需要額外處理

        # 檢查插值結果
        print("Interpolated shape:", new_ds.shape)
        print("Interpolated min:", new_ds.min().item(), "max:", new_ds.max().item())

        # 存儲新的插值數據
        output_filename = f'{name}_inter_{t}.nc'
        output_file = os.path.join(path_new, output_filename)
        new_ds.to_netcdf(output_file)

        print(f"Successfully saved {output_file}")



def nc_combine(file_paths, var, path_new, output_filename):
    # 定義一個空列表來儲存該文件里的.nc數據路徑
    dir_path = []
    
    for file_name in os.listdir(file_paths):
        dir_path.append(file_paths + file_name)
        print(dir_path)

    # dir_path ##输出文件路徑

    # 檢查每個文件的時間範圍並排序
    time_ranges = []
    for path in dir_path:
        ds = xr.open_dataset(path)
        time_values = ds.time.values
        time_ranges.append((path, time_values))
        ds.close()
    dir_path.sort(key=lambda x: time_ranges[[p[0] for p in time_ranges].index(x)][1])
    print("Sorted file paths: ", dir_path)

    print("success enter file...")

    data_new = []  ##建立一个空列表，存储逐日降雨数据
    for i in range(len(dir_path)):
        data = xr.open_dataset(dir_path[i])[var]
        data_new.append(data)  ##儲存数据
    da = xr.concat(data_new, dim='time')  ##将數據以時間维度来進行拼接
    # print(da)
    print("process...")

    # path_new = "C:/Users/miawu/nino/data/testing_data/" ##设置新路径
    # print(da.variable)
    da.to_netcdf(path_new + output_filename)  ##对拼接好的文件进行存储
    da.close()  ##关闭文件
    print(f"combine success to {output_filename}")

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
    """標準化數據並返回標準差"""
    ds = xr.open_dataset(file_path)
    print(ds)
    data = ds[var].values  # 形狀: (time, y, x)
    print("success loading data")

    # 計算均值和標準差（沿時間維度）
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)

    # 避免除以 0
    std[std == 0] = 1e-6

    print("calculating...")
    sst_normalized = (data - mean) / std

    if var == 'sss':
        # 創建新的 DataArray
        new_ds = xr.DataArray(sst_normalized,
                            dims=['n_model', 'n_mon', 'y', 'x'],
                            coords={'n_model': ds['n_model'], 'n_mon': ds['n_mon'], 'y': ds['y'], 'x': ds['x']},
                            name=out_var)
        
        # 保存標準差作為附加變數
        std_da = xr.DataArray(std,
                            dims=['n_model', 'y', 'x'],
                            coords={'n_model': ds['n_model'], 'y': ds['y'], 'x': ds['x']},
                            name=f'std{var}')

    elif var == 'temp':
        # 創建新的 DataArray
        new_ds = xr.DataArray(sst_normalized,
                            dims=['n_model', 'n_mon', 'lev', 'y', 'x'],
                            coords={'n_model': ds['n_model'], 'n_mon': ds['n_mon'], 'lev': ds['lev'], 'y': ds['y'], 'x': ds['x']},
                            name=out_var)
        
        # 保存標準差作為附加變數
        std_da = xr.DataArray(std,
                            dims=['n_model', 'lev', 'y', 'x'],
                            coords={'n_model': ds['n_model'], 'lev': ds['lev'], 'y': ds['y'], 'x': ds['x']},
                            name=f'std{var}')        
    
    # 合併數據和標準差
    output_ds = xr.Dataset({out_var: new_ds, f'std{var}': std_da})
    output_ds.to_netcdf(path_new + output_filename)
    print(f"success save data to {output_filename}")
    return output_ds

def calculate_nino34(ds, var, lat_name='y', lon_name='x'):
    """從 SST 數據中計算 Nino 3.4 指數"""
    # Nino 3.4 區域: 5°S-5°N, 170°W-120°W
    # nino_ds = ds[var][:,:,0,:,:].sel({lat_name: slice(-5, 5), lon_name: slice(-170, -120)})
    nino_ds = ds[var][:,:,0,:,:].sel({lat_name: slice(-5, 5), lon_name: slice(190, 240)})
    nino34 = nino_ds.mean(dim=[lat_name, lon_name], skipna=True)  # 空間平均，忽略 NaN
    return nino34

def merge(file_path_sss, file_path_sst, path_new, output_filename):
    # file_path_sss = path + "sss_norm_1850_2009.nc"
    # file_path_sst = path + "sst_norm_1850_2009.nc"
    ds_sss = xr.open_dataset(file_path_sss)
    ds_sst = xr.open_dataset(file_path_sst)
    ds_new = xr.merge([ds_sss, ds_sst])
    ds_new = ds_new.rename({'x': 'lon', 'y': 'lat'})
    ds_new.to_netcdf(path_new + output_filename)
    print(f"success merge data to {output_filename}")
    print(ds_new)

def merge_with_nino34(file_path_sss, file_path_sst, path_new, output_filename):
    """合併 SST、SSS、Nino 3.4 和標準化參數"""
    ds_sss = xr.open_dataset(file_path_sss)
    ds_sst = xr.open_dataset(file_path_sst)
    
    # 從 SST 數據中計算 Nino 3.4
    nino34 = calculate_nino34(ds_sst, 'temperatureNor')
    nino34_da = xr.DataArray(nino34, 
                            dims=['n_model', 'n_mon'], 
                            coords={'n_model': ds_sst['n_model'], 'n_mon': ds_sst['n_mon']}, 
                            name='nino34')
    
    # 合併所有數據
    ds_new = xr.merge([ds_sss, ds_sst, nino34_da])
    ds_new = ds_new.rename({'x': 'lon', 'y': 'lat'})
    ds_new.to_netcdf(path_new + output_filename)
    print(f"success merge data to {output_filename}")
    print(ds_new)

    return ds_new

def sperate_with_time(file_path, sel_time_start, sel_time_finish, path_new, output_filename):
    # file_path = "./data/testing_data/SST/tos_Omon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-203412.nc"
    ds = xr.open_dataset(file_path)
    print(ds)
    # print(ds["time"])
    # print(ds['tos'])
    ds_subset = ds.sel(n_mon=slice(sel_time_start, sel_time_finish))
    ds_subset.to_netcdf(path_new + output_filename)
    print(f"success sel time {sel_time_start} to {sel_time_finish}")

def expansion(file_path, input_var):
    ds = xr.open_dataset(file_path)
    print(ds)
    ds = ds.rename({'time': 'n_mon'})
    print(ds)

    # 假設只有一個模型、一個層
    n_model = 1
    lev = 1

    if input_var == 'tos':
        # expand dimensions
        sst_expanded = ds['tos'].expand_dims(dim={"n_model": [0], "lev": [0]})  # shape: (n_model, time, lev, lat, lon)
        sst_expanded = sst_expanded.transpose("n_model", "n_mon", "lev", "y", "x")


        # 組合新的 Dataset
        new_ds = xr.Dataset(
            {
                "temp": sst_expanded,

            },
            coords={
                "n_model": [0],
                "n_mon": ds.n_mon,
                "lev": [0],
                "lat": ds.lat,
                "lon": ds.lon
            }
        )

        # 儲存
        new_ds.to_netcdf("converted_dataset_sst_1850_2024.nc")

    elif input_var == 'sos':
        # expand dimensions
        sss_expanded = ds["sos"].expand_dims(dim={"n_model": [0]})  # shape: (n_model, time, lev, lat, lon)
        sss_expanded = sss_expanded.transpose("n_model", "n_mon", "y", "x")

        # 組合新的 Dataset
        new_ds = xr.Dataset(
            {
                "sss": sss_expanded,
                # "sssy": sss_expanded,

            },
            coords={
                "n_model": [0],
                "n_mon": ds.n_mon,
                "lat": ds.lat,
                "lon": ds.lon
            }
        )

        # 儲存
        new_ds.to_netcdf("converted_dataset_sss_1850_2024.nc")



def re_dataset(input_nc_file, output_dir, output_filename):
    # output_dir = './data/testing_data'
    # input_nc_file = "./data/testing_data/GFDL-CM4_sss_sst_2010_2024_1.nc"  # 輸入檔案路徑
    output_path = os.path.join(output_dir, output_filename)
    ds = xr.open_dataset(os.path.join(output_dir, input_nc_file))
    print(ds)

    temp = ds['temperatureNor'].isel(n_model=0)
    sss = ds['sssNor'].isel(n_model=0)
    stdtemp = ds['stdtemp'].isel(n_model=0)
    stdsss = ds['stdsss'].isel(n_model=0)
    nino34 = ds['nino34'].isel(n_model=0)

    temp_nor = xr.DataArray(temp, 
                            dims=['n_mon', 'lev', 'lat', 'lon'], 
                            coords={'n_mon': ds['n_mon'], 'lev': ds['lev'], 'lat': ds['lat'], 'lon': ds['lon']}, 
                            name='temperatureNor')

    sss_nor = xr.DataArray(sss, 
                            dims=['n_mon', 'lat', 'lon'], 
                            coords={'n_mon': ds['n_mon'], 'lat': ds['lat'], 'lon': ds['lon']}, 
                            name='sssNor')

    nino34 = xr.DataArray(nino34, 
                            dims=['n_mon'], 
                            coords={'n_mon': ds['n_mon']}, 
                            name='nino34')

    stdtemp = xr.DataArray(stdtemp, 
                            dims=['lev', 'lat', 'lon'], 
                            coords={'lev': ds['lev'], 'lat': ds['lat'], 'lon': ds['lon']}, 
                            name='stdtemp')
    stdsss = xr.DataArray(stdsss, 
                            dims=['lat', 'lon'], 
                            coords={'lat': ds['lat'], 'lon': ds['lon']}, 
                            name='stdsss')    
    # 合併所有數據
    ds_new = xr.merge([temp_nor, nino34, sss_nor, stdtemp, stdsss])
    print("dataset:\n")
    print(ds_new)
    # 儲存為 netCDF 檔
    ds_new.to_netcdf(output_path, format="NETCDF4", engine='netcdf4')


def all_data(output_dir, inter_dir, model_name):
    output_dir = "./data/all_data/"
    sss_dir = "./data/all_data/SSS/"
    sss_inter_dir = './data/all_data/sss_inter/'
    sst_inter_dir = './data/all_data/sst_inter/'

    nc_combine(sss_dir, 'sos', output_dir, f'{model_name}_sos_1850_2014.nc')

    # 1. 經度轉換(平移)
    sss_path = output_dir + f'{model_name}_sos_1850_2014.nc'

    out_sss = f'{model_name}_sos_convertlon_1850_2014.nc'
    convert_lon(sss_path, output_dir, out_sss)

    # 2. 補齊缺值
    sss_file = output_dir + 'sos_1850_2024_conlon.nc'
    sss_data = xr.open_dataset(sss_file)
    print(sss_data)
    sss_data_filled = sss_data.fillna(0.0)

    sss_data_filled.to_netcdf(os.path.join(output_dir, 'sos_1850_2024_conlon_1.nc'))
    ds_sss = xr.open_dataset(output_dir+'sos_1850_2024_conlon_1.nc')
    sss_ori = ds_sss['sos'].values
    print("sss_ori_region contains NaN:", np.any(np.isnan(sss_ori)))
    
    # 3. 線性插植
    sss_con = output_dir + 'sos_1850_2024_conlon_1.nc'
    interpolated(sss_con, 'sos', sss_inter_dir, 'sss')

    inter_out_sss = 'sss_inter_1850_2024.nc'
    nc_combine(sss_inter_dir, 'sos', output_dir, inter_out_sss)

    # 4. 計算異常值
    sss_interpolated_file = output_dir + "sss_inter_1850_2024.nc"
    anomaly(sss_interpolated_file, 'sos', '1961-01-16', '1990-12-16', output_dir, 'sss_anomaly_1850_2024.nc')
    
    # 5. 標準化 (Normalization)   從這裡開始執行
    sss_anomaly_file = output_dir + "sss_anomaly_1850_2024.nc"
    expansion(sss_anomaly_file, 'sos')

    sss_convert = './converted_dataset_sss_1850_2024.nc'
    ds = xr.open_dataset(sst_convert)
    print(ds)
    norm(sss_convert, 'sss', output_dir, 'sss_norm_1850_2024.nc', 'sssNor')
    
    # 6. 合併 SST 和 SSS 數據
    sss_norm_file = output_dir + "sss_norm_1850_2024.nc"
    merge_with_nino34(sss_norm_file, sst_norm_file, output_dir, 'GFDL-CM4_sss_sst_nino_1850_2024.nc')
    merge(sss_norm_file, sst_norm_file, output_dir, 'GFDL-CM4_sss_sst_1850_2024.nc')

    print("all_data process successful!")


# 執行流程
if __name__ == "__main__":
    path = "./data/training_data/"
    path_test = './data/testing_data/'
    path_all = './data/all_data/'
    sss_norm_file = './data/testing_data/sss_norm_2010_2024.nc'
    sst_norm_file = './data/testing_data/sst_norm_2010_2024.nc'

    all_data()
    sperate_with_time("./data/all_data/GFDL-CM4_sss_sst_nino_1850_2024.nc", '1850-01', '1989-12', path_all, 'GFDL-CM4_sss_sst_1850_1989.nc')
    sperate_with_time("./data/all_data/GFDL-CM4_sss_sst_nino_1850_2024.nc", '1990-01', '2009-12', path_all, 'GFDL-CM4_sss_sst_1990_2009.nc')
    train_data_speration()
    # re_dataset("GFDL-CM4_sss_sst_nino_2010_2024.nc", path_all, "GFDL-CM4_sss_sst_nino_2010_2024_1.nc")
    # test_data_speration("GFDL-CM4_sss_sst_nino_2010_2024.nc", path_all, "GFDL-CM4_sss_sst_2010_2024_1.nc")

import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
# import netCDF4 as nc
import xarray as xr
from scipy.interpolate import griddata
import os
import pandas as pd


def plot_dataset(dataset: xr.Dataset):
    data = dataset.copy()
    projection = ccrs.Mercator()
    crs = ccrs.PlateCarree()
    plt.figure(figsize=(16, 9), dpi=150)
    ax = plt.axes(projection=projection, frameon=True)

    gl = ax.gridlines(crs=crs, draw_labels=True,
                      linewidth=.6, color='gray', alpha=0.5, linestyle='-.')
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    # To plot borders and coastlines, we can use cartopy feature
    ax.add_feature(cf.COASTLINE.with_scale("50m"), lw=0.5)
    ax.add_feature(cf.BORDERS.with_scale("50m"), lw=0.3)

    # Now, we will specify extent of our map in minimum/maximum longitude/latitude
    # Note that these values are specified in degrees of longitude and degrees of latitude
    # However, we can specify them in any crs that we want, but we need to provide appropriate
    # crs argument in ax.set_extent
    # crs is PlateCarree -> we are explicitly telling axes, that we are creating bounds that are in degrees
    lon_min = -299
    lon_max = 60
    lat_min = -55
    lat_max = 60

    ##### WE ADDED THESE LINES #####
    cbar_kwargs = {'orientation': 'horizontal', 'shrink': 0.6, "pad": .05, 'aspect': 40,
                   'label': 'sea surface temperature'}
    data['sssNor_out'].isel(group=0, Tout=0).plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs, levels=21, vmin=-1, vmax=1, cmap='RdBu_r')
    ################################

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)
    # plt.title(f"Temperature anomaly over Europe in {dataset.valid_time.dt.strftime('%B %Y').values}")
    plt.show()

def convert_lon(file_path, out_dir, output_filename):
    ds = xr.open_dataset(file_path)
    print(ds)
    # 轉換經度範圍到 0~360
    ds["x"] = (ds["x"] + 360) % 360

    # 排序經度（由小到大）
    ds = ds.sortby("x")
    print(ds)

    output_file = os.path.join(out_dir, output_filename)
    ds.to_netcdf(output_file)
    print(f"success save {output_filename}")

def interpolated(file_path, var, path_new, name):
    # 加載數據
    # ds, var_name = interpolated(file_path)  # 假設上面定義了檢查函數
    print("=======================")
    print("loading data...")
    ds = xr.open_dataset(file_path)
    print("Dataset structure:")
    print(ds)
    shape = ds[var].values.shape
    print(shape)
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

    # # 平移經度到 -180° 到 180°
    # shift = int((360 - 60.3) / 0.25)  # 計算需要平移的格點數
    # ds = ds.roll(x=-shift, roll_coords=True)
    # ds['x'] = (ds['x'] + 360) % 360  # 將經度轉換到 -180° 到 180°


    # 目標插值網格（5° x 5°）
    # 緯度區域設定：高解析赤道區
    lat_south = np.arange(-20, -5, 1.0)         # -20, -19, ..., -6
    lat_equator = np.arange(-5, 5.5, 0.5)       # -5.0, -4.5, ..., 5.0
    lat_north = np.arange(6, 21, 1.0)           # 6, 7, ..., 20

    # 合併為非均勻解析度的緯度陣列
    new_lat = np.concatenate([lat_south, lat_equator, lat_north])

    # new_lat = np.arange(-20, 21, 1.0)  # 24 個點
    new_lon = np.arange(2, 359, 2.0)   # 72 個點
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
    new_lat_2d_interp = griddata(points, values_lat, xi, method='linear').reshape(179, 51)
    new_lon_2d_interp = griddata(points, values_lon, xi, method='linear').reshape(179, 51)

    print("new_lat_2d_interp shape:", new_lat_2d_interp.shape)
    print("new_lon_2d_interp shape:", new_lon_2d_interp.shape)
    print("new_lat_2d_interp min:", new_lat_2d_interp.min(), "max:", new_lat_2d_interp.max())
    print("new_lon_2d_interp min:", new_lon_2d_interp.min(), "max:", new_lon_2d_interp.max())

    # 創建臨時坐標映射
    print("start interpolated...")
    ds_temp = ds.assign_coords(lat=('y', lat_2d.mean(axis=1)), lon=('x', lon_2d.mean(axis=0)))

    for t in range(shape[0]):
        new_ds = ds_temp[var].isel(time=t).interp(x=new_lon, y=new_lat, method='linear', kwargs={'fill_value': None})

        # 分配插值後的 2D 坐標
        new_ds = new_ds.assign_coords(lat=('y', new_lat_2d_interp[0, :]), lon=('x', new_lon_2d_interp[:, 0]))
        # 注意：這裡只分配 1D 近似，因為 assign_coords 期望每個維度對應一個 1D 陣列
        # 如果需要 2D 坐標，可以使用多維坐標，但需要額外處理

        # 檢查插值結果
        print("Interpolated shape:", new_ds.shape)
        print("Interpolated min:", new_ds.min().item(), "max:", new_ds.max().item())

        # 存儲新的插值數據
        output_filename = f'{name}_inter_{t}.nc'
        output_file = os.path.join(path_new, output_filename)
        new_ds.to_netcdf(output_file)

        print(f"Successfully saved {output_file}")



def nc_combine(file_paths, var, path_new, output_filename):
    # 定義一個空列表來儲存該文件里的.nc數據路徑
    dir_path = []
    
    for file_name in os.listdir(file_paths):
        dir_path.append(file_paths + file_name)
        print(dir_path)

    # dir_path ##输出文件路徑

    # 檢查每個文件的時間範圍並排序
    time_ranges = []
    for path in dir_path:
        ds = xr.open_dataset(path)
        time_values = ds.time.values
        time_ranges.append((path, time_values))
        ds.close()
    dir_path.sort(key=lambda x: time_ranges[[p[0] for p in time_ranges].index(x)][1])
    print("Sorted file paths: ", dir_path)

    print("success enter file...")

    data_new = []  ##建立一个空列表，存储逐日降雨数据
    for i in range(len(dir_path)):
        data = xr.open_dataset(dir_path[i])[var]
        data_new.append(data)  ##儲存数据
    da = xr.concat(data_new, dim='time')  ##将數據以時間维度来進行拼接
    # print(da)
    print("process...")

    # path_new = "C:/Users/miawu/nino/data/testing_data/" ##设置新路径
    # print(da.variable)
    da.to_netcdf(path_new + output_filename)  ##对拼接好的文件进行存储
    da.close()  ##关闭文件
    print(f"combine success to {output_filename}")

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
    """標準化數據並返回標準差"""
    ds = xr.open_dataset(file_path)
    print(ds)
    data = ds[var].values  # 形狀: (time, y, x)
    print("success loading data")

    # 計算均值和標準差（沿時間維度）
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)

    # 避免除以 0
    std[std == 0] = 1e-6

    print("calculating...")
    sst_normalized = (data - mean) / std

    if var == 'sss':
        # 創建新的 DataArray
        new_ds = xr.DataArray(sst_normalized,
                            dims=['n_model', 'n_mon', 'y', 'x'],
                            coords={'n_model': ds['n_model'], 'n_mon': ds['n_mon'], 'y': ds['y'], 'x': ds['x']},
                            name=out_var)
        
        # 保存標準差作為附加變數
        std_da = xr.DataArray(std,
                            dims=['n_model', 'y', 'x'],
                            coords={'n_model': ds['n_model'], 'y': ds['y'], 'x': ds['x']},
                            name=f'std{var}')

    elif var == 'temp':
        # 創建新的 DataArray
        new_ds = xr.DataArray(sst_normalized,
                            dims=['n_model', 'n_mon', 'lev', 'y', 'x'],
                            coords={'n_model': ds['n_model'], 'n_mon': ds['n_mon'], 'lev': ds['lev'], 'y': ds['y'], 'x': ds['x']},
                            name=out_var)
        
        # 保存標準差作為附加變數
        std_da = xr.DataArray(std,
                            dims=['n_model', 'lev', 'y', 'x'],
                            coords={'n_model': ds['n_model'], 'lev': ds['lev'], 'y': ds['y'], 'x': ds['x']},
                            name=f'std{var}')        
    
    # 合併數據和標準差
    output_ds = xr.Dataset({out_var: new_ds, f'std{var}': std_da})
    output_ds.to_netcdf(path_new + output_filename)
    print(f"success save data to {output_filename}")
    return output_ds

def calculate_nino34(ds, var, lat_name='y', lon_name='x'):
    """從 SST 數據中計算 Nino 3.4 指數"""
    # Nino 3.4 區域: 5°S-5°N, 170°W-120°W
    # nino_ds = ds[var][:,:,0,:,:].sel({lat_name: slice(-5, 5), lon_name: slice(-170, -120)})
    nino_ds = ds[var][:,:,0,:,:].sel({lat_name: slice(-5, 5), lon_name: slice(190, 240)})
    nino34 = nino_ds.mean(dim=[lat_name, lon_name], skipna=True)  # 空間平均，忽略 NaN
    return nino34

def merge(file_path_sss, file_path_sst, path_new, output_filename):
    # file_path_sss = path + "sss_norm_1850_2009.nc"
    # file_path_sst = path + "sst_norm_1850_2009.nc"
    ds_sss = xr.open_dataset(file_path_sss)
    ds_sst = xr.open_dataset(file_path_sst)
    ds_new = xr.merge([ds_sss, ds_sst])
    ds_new = ds_new.rename({'x': 'lon', 'y': 'lat'})
    ds_new.to_netcdf(path_new + output_filename)
    print(f"success merge data to {output_filename}")
    print(ds_new)

def merge_with_nino34(file_path_sss, file_path_sst, path_new, output_filename):
    """合併 SST、SSS、Nino 3.4 和標準化參數"""
    ds_sss = xr.open_dataset(file_path_sss)
    ds_sst = xr.open_dataset(file_path_sst)
    
    # 從 SST 數據中計算 Nino 3.4
    nino34 = calculate_nino34(ds_sst, 'temperatureNor')
    nino34_da = xr.DataArray(nino34, 
                            dims=['n_model', 'n_mon'], 
                            coords={'n_model': ds_sst['n_model'], 'n_mon': ds_sst['n_mon']}, 
                            name='nino34')
    
    # 合併所有數據
    ds_new = xr.merge([ds_sss, ds_sst, nino34_da])
    ds_new = ds_new.rename({'x': 'lon', 'y': 'lat'})
    ds_new.to_netcdf(path_new + output_filename)
    print(f"success merge data to {output_filename}")
    print(ds_new)

    return ds_new

def sperate_with_time(file_path, sel_time_start, sel_time_finish, path_new, output_filename):
    # file_path = "./data/testing_data/SST/tos_Omon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-203412.nc"
    ds = xr.open_dataset(file_path)
    print(ds)
    # print(ds["time"])
    # print(ds['tos'])
    ds_subset = ds.sel(n_mon=slice(sel_time_start, sel_time_finish))
    ds_subset.to_netcdf(path_new + output_filename)
    print(f"success sel time {sel_time_start} to {sel_time_finish}")

def expansion(file_path, input_var):
    ds = xr.open_dataset(file_path)
    print(ds)
    ds = ds.rename({'time': 'n_mon'})
    print(ds)

    # 假設只有一個模型、一個層
    n_model = 1
    lev = 1

    if input_var == 'tos':
        # expand dimensions
        sst_expanded = ds['tos'].expand_dims(dim={"n_model": [0], "lev": [0]})  # shape: (n_model, time, lev, lat, lon)
        sst_expanded = sst_expanded.transpose("n_model", "n_mon", "lev", "y", "x")


        # 組合新的 Dataset
        new_ds = xr.Dataset(
            {
                "temp": sst_expanded,

            },
            coords={
                "n_model": [0],
                "n_mon": ds.n_mon,
                "lev": [0],
                "lat": ds.lat,
                "lon": ds.lon
            }
        )

        # 儲存
        new_ds.to_netcdf("converted_dataset_sst_1850_2024.nc")

    elif input_var == 'sos':
        # expand dimensions
        sss_expanded = ds["sos"].expand_dims(dim={"n_model": [0]})  # shape: (n_model, time, lev, lat, lon)
        sss_expanded = sss_expanded.transpose("n_model", "n_mon", "y", "x")

        # 組合新的 Dataset
        new_ds = xr.Dataset(
            {
                "sss": sss_expanded,
                # "sssy": sss_expanded,

            },
            coords={
                "n_model": [0],
                "n_mon": ds.n_mon,
                "lat": ds.lat,
                "lon": ds.lon
            }
        )

        # 儲存
        new_ds.to_netcdf("converted_dataset_sss_1850_2024.nc")



def re_dataset(input_nc_file, output_dir, output_filename):
    # output_dir = './data/testing_data'
    # input_nc_file = "./data/testing_data/GFDL-CM4_sss_sst_2010_2024_1.nc"  # 輸入檔案路徑
    output_path = os.path.join(output_dir, output_filename)
    ds = xr.open_dataset(os.path.join(output_dir, input_nc_file))
    print(ds)

    temp = ds['temperatureNor'].isel(n_model=0)
    sss = ds['sssNor'].isel(n_model=0)
    stdtemp = ds['stdtemp'].isel(n_model=0)
    stdsss = ds['stdsss'].isel(n_model=0)
    nino34 = ds['nino34'].isel(n_model=0)

    temp_nor = xr.DataArray(temp, 
                            dims=['n_mon', 'lev', 'lat', 'lon'], 
                            coords={'n_mon': ds['n_mon'], 'lev': ds['lev'], 'lat': ds['lat'], 'lon': ds['lon']}, 
                            name='temperatureNor')

    sss_nor = xr.DataArray(sss, 
                            dims=['n_mon', 'lat', 'lon'], 
                            coords={'n_mon': ds['n_mon'], 'lat': ds['lat'], 'lon': ds['lon']}, 
                            name='sssNor')

    nino34 = xr.DataArray(nino34, 
                            dims=['n_mon'], 
                            coords={'n_mon': ds['n_mon']}, 
                            name='nino34')

    stdtemp = xr.DataArray(stdtemp, 
                            dims=['lev', 'lat', 'lon'], 
                            coords={'lev': ds['lev'], 'lat': ds['lat'], 'lon': ds['lon']}, 
                            name='stdtemp')
    stdsss = xr.DataArray(stdsss, 
                            dims=['lat', 'lon'], 
                            coords={'lat': ds['lat'], 'lon': ds['lon']}, 
                            name='stdsss')    
    # 合併所有數據
    ds_new = xr.merge([temp_nor, nino34, sss_nor, stdtemp, stdsss])
    print("dataset:\n")
    print(ds_new)
    # 儲存為 netCDF 檔
    ds_new.to_netcdf(output_path, format="NETCDF4", engine='netcdf4')


def all_data(output_dir, inter_dir, model_name):
    output_dir = "./data/all_data/"
    sss_dir = "./data/all_data/SSS/"
    sss_inter_dir = './data/all_data/sss_inter/'
    sst_inter_dir = './data/all_data/sst_inter/'

    nc_combine(sss_dir, 'sos', output_dir, f'{model_name}_sos_1850_2014.nc')

    # 1. 經度轉換(平移)
    sss_path = output_dir + f'{model_name}_sos_1850_2014.nc'

    out_sss = f'{model_name}_sos_convertlon_1850_2014.nc'
    convert_lon(sss_path, output_dir, out_sss)

    # 2. 補齊缺值
    sss_file = output_dir + 'sos_1850_2024_conlon.nc'
    sss_data = xr.open_dataset(sss_file)
    print(sss_data)
    sss_data_filled = sss_data.fillna(0.0)

    sss_data_filled.to_netcdf(os.path.join(output_dir, 'sos_1850_2024_conlon_1.nc'))
    ds_sss = xr.open_dataset(output_dir+'sos_1850_2024_conlon_1.nc')
    sss_ori = ds_sss['sos'].values
    print("sss_ori_region contains NaN:", np.any(np.isnan(sss_ori)))
    
    # 3. 線性插植
    sss_con = output_dir + 'sos_1850_2024_conlon_1.nc'
    interpolated(sss_con, 'sos', sss_inter_dir, 'sss')

    inter_out_sss = 'sss_inter_1850_2024.nc'
    nc_combine(sss_inter_dir, 'sos', output_dir, inter_out_sss)

    # 4. 計算異常值
    sss_interpolated_file = output_dir + "sss_inter_1850_2024.nc"
    anomaly(sss_interpolated_file, 'sos', '1961-01-16', '1990-12-16', output_dir, 'sss_anomaly_1850_2024.nc')
    
    # 5. 標準化 (Normalization)   從這裡開始執行
    sss_anomaly_file = output_dir + "sss_anomaly_1850_2024.nc"
    expansion(sss_anomaly_file, 'sos')

    sss_convert = './converted_dataset_sss_1850_2024.nc'
    ds = xr.open_dataset(sst_convert)
    print(ds)
    norm(sss_convert, 'sss', output_dir, 'sss_norm_1850_2024.nc', 'sssNor')
    
    # 6. 合併 SST 和 SSS 數據
    sss_norm_file = output_dir + "sss_norm_1850_2024.nc"
    merge_with_nino34(sss_norm_file, sst_norm_file, output_dir, 'GFDL-CM4_sss_sst_nino_1850_2024.nc')
    merge(sss_norm_file, sst_norm_file, output_dir, 'GFDL-CM4_sss_sst_1850_2024.nc')

    print("all_data process successful!")


# 執行流程
if __name__ == "__main__":
    cmip6_dir = './cmip6_sos_data/'

    AWI_dir = './cmip6_sos_data/AWI-CM-1-1-MR/'
    CESM2_dir = './cmip6_sos_data/CESM2/'
    CESM2_WACCM_FV2_dir = './cmip6_sos_data/CESM2-WACCM-FV2/'
    EC_Earth3_CC_dir = './cmip6_sos_data/EC-Earth3-CC/'
    FGOALS_g3_dir = './cmip6_sos_data/FGOALS-g3/'
    GFDL_CM4_dir = './cmip6_sos_data/GFDL-CM4/'
    GFDL_ESM4_dir = './cmip6_sos_data/GFDL-ESM4/'
    GISS_E2_1_G_dir = './cmip6_sos_data/GISS-E2-1-G/'
    GISS_E2_1_H_dir = './cmip6_sos_data/GISS-E2-1-H/'
    MIROC6_dir = './cmip6_sos_data/MIROC6/'
    MPI_ESM1_2_LR_dir = './cmip6_sos_data/MPI-ESM1-2-LR/'
    MPI_ESM_1_2_HAM_dir = './cmip6_sos_data/MPI-ESM-1-2-HAM/'
    NorESM2_MM_dir = './cmip6_sos_data/NorESM2-MM/'
    SAM0_UNICON_dir = './cmip6_sos_data/SAM0-UNICON/'
    CMCC_CM2_HR4_dir = './cmip6_sos_data/CMCC-CM2-HR4/'

    nc_combine(AWI_dir, 'sos', cmip6_dir, 'AWI-CM-1-1-MR_185001-201412.nc')
    nc_combine(CESM2_dir, 'sos', cmip6_dir, 'CESM2_185001_201412.nc')
    nc_combine(CESM2_WACCM_FV2_dir, 'sos', cmip6_dir, 'CESM2-WACCM-FV2_185001-201412.nc')
    nc_combine(EC_Earth3_CC_dir, 'sos', cmip6_dir, 'EC-Earth3-CC_185001-201412.nc')
    nc_combine(FGOALS_g3_dir, 'sos', cmip6_dir, 'FGOALS-g3_185001-201412.nc')
    nc_combine(GFDL_CM4_dir, 'sos', cmip6_dir, 'GFDL-CM4_185001-201412.nc')
    nc_combine(GFDL_ESM4_dir, 'sos', cmip6_dir, 'GFDL-ESM4_185001-201412.nc')
    nc_combine(GISS_E2_1_G_dir, 'sos', cmip6_dir, 'GISS-E2-1-G_185001-201412.nc')
    nc_combine(GISS_E2_1_H_dir, 'sos', cmip6_dir, 'GISS-E2-1-H_185001-201412.nc')
    nc_combine(MIROC6_dir, 'sos', cmip6_dir, 'MIROC6_185001-201412.nc')
    nc_combine(MPI_ESM1_2_LR_dir, 'sos', cmip6_dir, 'MPI-ESM1-2-LR_185001-201412.nc')
    nc_combine(MPI_ESM_1_2_HAM_dir, 'sos', cmip6_dir, 'MPI-ESM-1-2-HAM_185001-201412.nc')
    nc_combine(NorESM2_MM_dir, 'sos', cmip6_dir, 'NorESM2-MM_185001-201412.nc')
    nc_combine(SAM0_UNICON_dir, 'sos', cmip6_dir, 'SAM0-UNICON_185001-201412.nc')
    nc_combine(CMCC_CM2_HR4_dir, 'sos', cmip6_dir, 'CMCC-CM2-HR4_185001-201412.nc')
    
    convert_lon('cmip6_sos_data/AWI-CM-1-1-MR_185001-201412.nc', cmip6_dir, 'AWI-CM-1-1-MR_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/CESM2_185001-201412.nc', cmip6_dir, 'CESM2_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/EC-Earth3-CC_185001-201412.nc', cmip6_dir, 'EC-Earth3-CC_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/FGOALS-g3_185001-201412.nc', cmip6_dir, 'FGOALS-g3_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/GFDL-CM4_185001-201412.nc', cmip6_dir, 'GFDL-CM4_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/GFDL-ESM4_185001-201412.nc', cmip6_dir, 'GFDL-ESM4_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/GISS-E2-1-G_185001-201412.nc', cmip6_dir, 'GISS-E2-1-G_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/GISS-E2-1-H_185001-201412.nc', cmip6_dir, 'GISS-E2-1-H_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/MIROC6_185001-201412.nc', cmip6_dir, 'MIROC6_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/MPI-ESM1-2-LR_185001-201412.nc', cmip6_dir, 'MPI-ESM1-2-LR_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/MPI-ESM-1-2-HAM_185001-201412.nc', cmip6_dir, 'MPI-ESM-1-2-HAM_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/NorESM2-MM_185001-201412.nc', cmip6_dir, 'NorESM2-MM_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/SAM0-UNICON_185001-201412.nc', cmip6_dir, 'SAM0-UNICON_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/CMCC-CM2-HR4_185001-201412.nc', cmip6_dir, 'CMCC-CM2-HR4_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/sos_Omon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc', cmip6_dir, 'ACCESS-CM2_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/sos_Omon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc', cmip6_dir, 'ACCESS-ESM1-5_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/sos_Omon_CanESM5-CanOE_historical_r1i1p2f1_gn_185001-201412.nc', cmip6_dir, 'CanESM5-CanOE_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/sos_Omon_CESM2-WACCM_historical_r1i1p1f1_gn_185001-201412.nc', cmip6_dir, 'CESM2-WACCM_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/sos_Omon_FGOALS-f3-L_historical_r2i1p1f1_gn_185001-201412.nc', cmip6_dir, 'FGOALS-f3-L_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/sos_Omon_IPSL-CM6A-LR_historical_r1i1p1f1_gn_185001-201412.nc', cmip6_dir, 'IPSL-CM6A-LR_185001-201412_convert.nc')
    convert_lon('cmip6_sos_data/sos_Omon_NESM3_historical_r1i1p1f1_gn_185001-201412.nc', cmip6_dir, 'NESM3_185001-201412_convert.nc')
    # all_data()
    # sperate_with_time("./data/all_data/GFDL-CM4_sss_sst_nino_1850_2024.nc", '1850-01', '1989-12', path_all, 'GFDL-CM4_sss_sst_1850_1989.nc')
    # sperate_with_time("./data/all_data/GFDL-CM4_sss_sst_nino_1850_2024.nc", '1990-01', '2009-12', path_all, 'GFDL-CM4_sss_sst_1990_2009.nc')
    # train_data_speration()
    # re_dataset("GFDL-CM4_sss_sst_nino_2010_2024.nc", path_all, "GFDL-CM4_sss_sst_nino_2010_2024_1.nc")
    # test_data_speration("GFDL-CM4_sss_sst_nino_2010_2024.nc", path_all, "GFDL-CM4_sss_sst_2010_2024_1.nc")



