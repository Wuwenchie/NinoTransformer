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
from tqdm import tqdm  # 顯示進度條


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
    data['sos'].isel(time=0).plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs, levels=21, vmin=-1, vmax=1, cmap='RdBu_r')
    ################################

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)
    # plt.title(f"Temperature anomaly over Europe in {dataset.valid_time.dt.strftime('%B %Y').values}")
    plt.show()



def convert_lon(file_path, out_dir, output_filename):
    ds = xr.open_dataset(file_path)
    print(file_path)
    print("Before conversion:", ds)

    # lon_name = input("print the longitude name:")
    lon_value = ds['i'].values
    print(lon_value.min(), lon_value.max())

    # 經度轉換到 0~360
    ds['i'] = (ds['i'] + 360.0) % 360.0
    ds['j'] = (ds['j'] + 90.0) % 90.0

    # 經度排序（從小到大）
    ds = ds.sortby('i')
    ds = ds.sortby('j')

    # 將 n_mon 變成數字 index（0 ~ 1979）185001~201412
    ds = ds.rename({'time': 'n_mon'})
    ds = ds.assign_coords(n_mon=np.arange(ds.sizes["n_mon"]))

    print("After conversion:", ds)

    # 儲存輸出
    output_file = os.path.join(out_dir, output_filename)
    ds.to_netcdf(output_file)
    print(f"成功儲存：{output_filename}")

def convert_time(input_dir, input_filename, output_dir, output_filename):
    input_nc_file = os.path.join(input_dir, input_filename)
    output_nc_file = os.path.join(output_dir, output_filename)
    
    ds = xr.open_dataset(input_nc_file)
    print("Before conversion:", ds)
    ds = ds.rename({'time': 'n_mon'})
    ds = ds.assign_coords(n_mon=np.arange(ds.sizes["n_mon"]))
    print("After conversion:", ds)
    ds.to_netcdf(output_nc_file)
    print(f"成功儲存：{output_filename}")



def interpolate_to_regular_grid(file_path, output_dir, output_prefix, var_name='sos',
                                lon_res=1.0, lat_res=1.0,
                                lon_range=(0, 360), lat_range=(-90, 90)):
    print("Opening dataset...")
    ds = xr.open_dataset(file_path)
    data = ds[var_name]   # (time, i, j)
    lat_2d = ds['latitude'].values  # (i, j)
    lon_2d = ds['longitude'].values # (i, j)

    print(f"Input variable shape: {data.shape}")
    print(f"Latitude range: {lat_2d.min()} to {lat_2d.max()}")
    print(f"Longitude range: {lon_2d.min()} to {lon_2d.max()}")

    # 建立規則網格
    new_lats = np.arange(lat_range[0], lat_range[1] + lat_res, lat_res)
    new_lons = np.arange(lon_range[0], lon_range[1], lon_res)
    lon_grid, lat_grid = np.meshgrid(new_lons, new_lats)  # (lat, lon)

    interp_points = np.column_stack((lon_2d.ravel(), lat_2d.ravel()))

    # 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)

    print("Interpolating over time steps...")

    for t in tqdm(range(data.shape[0])):
        field = data.isel(time=t).values  # shape (i, j)
        field_flat = field.ravel()

        # 插值
        interp_values = griddata(interp_points, field_flat,
                                 (lon_grid, lat_grid), method='linear')

        # 儲存成 xarray 結構
        out_ds = xr.Dataset(
            {
                var_name: (("lat", "lon"), interp_values.astype(np.float32))
            },
            coords={
                "lat": new_lats,
                "lon": new_lons,
                "time": ds['time'].isel(time=t)
            }
        )

        # 儲存 NetCDF
        out_file = os.path.join(output_dir, f"{output_prefix}_interp_{t:04d}.nc")
        out_ds.to_netcdf(out_file)

    print(f"✅ 完成插值與儲存，共 {data.shape[0]} 筆時間步。結果保存在: {output_dir}")


def interpolated(file_path, path_new, name):
    print(f"Opening dataset of {file_path}")
    print("Loading data...")
    ds = xr.open_dataset(file_path)
    ds =ds.fillna(0.0)
    print("Dataset structure:")
    print(ds)
    print("=======================")

    shape = ds['sos'].shape
    print("sos shape:", shape)

    # 處理經緯度：檢查是 1D 還是 2D
    if ds['lat'].ndim == 1 and ds['lon'].ndim == 1:
        lat_1d = ds['lat'].values
        lon_1d = ds['lon'].values
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    else:
        lat_2d = ds['lat'].values
        lon_2d = ds['lon'].values

    print("lat_2d shape:", lat_2d.shape)
    print("lon_2d shape:", lon_2d.shape)

    # 檢查坐標維度
    if lat_2d.shape != lon_2d.shape:
        raise ValueError("Latitude and longitude grids must have the same shape")

    # 設定插值後目標緯度（非均勻解析度）
    lat_south = np.arange(-20, -5, 1.0)
    lat_equator = np.arange(-5, 5.5, 0.5)
    lat_north = np.arange(6, 21, 1.0)
    new_lat = np.concatenate([lat_south, lat_equator, lat_north])
    new_lon = np.arange(2, 359, 2.0)

    new_lon_2d, new_lat_2d = np.meshgrid(new_lon, new_lat)
    print("new_lat_2d shape:", new_lat_2d.shape)
    print("new_lon_2d shape:", new_lon_2d.shape)

    # 插值點準備
    print("Preparing interpolation grid...")
    points = np.column_stack((lon_2d.ravel(), lat_2d.ravel()))
    xi = np.column_stack((new_lon_2d.ravel(), new_lat_2d.ravel()))

    # 插值每個時間步
    for t in tqdm(range(shape[0])):
        # print(f"Interpolating time step {t}...")
        sos_data = ds['sos'].isel(time=t).values.ravel()
        # sos_data = ds['sos'].isel(n_mon=t).values.ravel()
        sos_interp = griddata(points, sos_data, xi, method='linear')
        sos_interp_2d = sos_interp.reshape(new_lat_2d.shape)

        # 建立新的 DataArray 和 Dataset
        da = xr.DataArray(
            sos_interp_2d,
            dims=('lat', 'lon'),
            coords={'lat': ('lat', new_lat), 'lon': ('lon', new_lon)},
            name='sos'
        )
        # da = da.expand_dims(n_mon=[ds['n_mon'].values[t]])
        # print(da)
        da = da.expand_dims(time=[ds['time'].values[t]])

        new_ds = xr.Dataset({'sos': da})

        # 儲存
        output_filename = f'{name}_inter_{t}.nc'
        output_file = os.path.join(path_new, output_filename)
        new_ds.to_netcdf(output_file)

        # print(f"Saved: {output_file}")
    print(f"✅ 完成插值與儲存，共 {shape[0]} 筆時間步。結果保存在: {path_new}")
    

def nc_combine(file_paths, var, path_new, output_filename):
    # 定義一個空列表來儲存該文件里的.nc數據路徑
    dir_path = []
    
    for file_name in os.listdir(file_paths):
        dir_path.append(os.path.join(file_paths, file_name))
        print(dir_path)

    # dir_path ##输出文件路徑

    # 檢查每個文件的時間範圍並排序
    time_ranges = []
    for path in dir_path:
        ds = xr.open_dataset(path)
        # print(ds)
        time_values = ds.time.values
        # time_values = ds.n_mon.values
        time_ranges.append((path, time_values))
        ds.close()
    dir_path.sort(key=lambda x: time_ranges[[p[0] for p in time_ranges].index(x)][1])
    print("Sorted file paths: ", dir_path)

    print("success enter file...")

    data_new = []  ##建立一个空列表，存储逐日数据
    for i in range(len(dir_path)):
        data = xr.open_dataset(dir_path[i])[var]
        data_new.append(data)  ##儲存数据
    da = xr.concat(data_new, dim='time')  ##将數據以時間维度来進行拼接
    # da = xr.concat(data_new, dim='n_mon')  ##将數據以時間维度来進行拼接
    # print(da)
    print("process...")

    # path_new = "C:/Users/miawu/nino/data/testing_data/" ##设置新路径
    # print(da.variable)
    da = da.rename({'time': 'n_mon'})
    da = da.assign_coords(n_mon=np.arange(da.sizes["n_mon"]))
    da.to_netcdf(os.path.join(path_new, output_filename))  ##对拼接好的文件进行存储
    print("After combine:",da)
    da.close()  ##关闭文件
    print(f"combine success to {output_filename}")

def model_combine(input_dir, output_dir, output_filename):
    # 定義一個空列表來儲存該文件里的.nc數據路徑
    dir_path = []
    
    for file_name in os.listdir(input_dir):
        dir_path.append(os.path.join(input_dir, file_name))
        print(dir_path)

    # dir_path ##输出文件路徑

    # 檢查每個文件的時間範圍並排序
    model_ranges = []
    for path in dir_path:
        ds = xr.open_dataset(path)
        # print(ds)
        model_values = ds.n_model.values
        # time_values = ds.n_mon.values
        model_ranges.append((path, model_values))
        ds.close()
    dir_path.sort(key=lambda x: model_ranges[[p[0] for p in model_ranges].index(x)][1])
    print("Sorted file paths: ", dir_path)

    print("success enter file...")

    data_new = []  ##建立一个空列表，存储逐日数据
    for i in range(len(dir_path)):
        data_sss = xr.open_dataset(dir_path[i])['sssNor']
        data_std = xr.open_dataset(dir_path[i])['stdsss']
        data = xr.merge([data_sss, data_std])
        data_new.append(data)  ##儲存数据
    da = xr.concat(data_new, dim='n_model')  ##将數據以時間维度来進行拼接
    print("process...")

    da.to_netcdf(os.path.join(output_dir, output_filename))  ##对拼接好的文件进行存储
    print("After combine:",da)
    da.close()  ##关闭文件
    print(f"combine success to {output_filename}")

def anomaly(file_path, out_dir, output_filename):
    # 讀取 NetCDF SST 數據
    ds = xr.open_dataset(file_path)
    print(ds)
    data = ds['sos']  # 假設 SST 形狀為 (time, lat, lon)

    print("successful loading data..")

    # 計算月氣候平均值 (1981年-2010年 = n_mon:1583-1931)
    climatology = data.sel(n_mon=slice('1583', '1931')).groupby("n_mon").mean(dim="n_mon")
    print("successful loading data...")

    # 去除趨勢
    # print("start detrend...")
    # data_detrended = xr.apply_ufunc(detrend, data, axis=0)
    # print("success")

    # 計算距平
    print("start calculate anomalies...")
    anomalies = data.groupby("n_mon") - climatology
    print("sucess")

    # 保存結果
    anomalies.to_netcdf(os.path.join(out_dir, output_filename))
    print(f"successful save {output_filename}")

def norm(file_path, out_dir, output_filename):
    """標準化數據並返回標準差"""
    ds = xr.open_dataset(file_path)
    print(ds)
    data = ds['sss'].values  # 形狀: (time, y, x)
    print("success loading data")

    # 計算均值和標準差（沿時間維度）
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)

    # 避免除以 0
    std[std == 0] = 1e-6

    print("calculating...")
    sst_normalized = (data - mean) / std

    # 創建新的 DataArray
    new_ds = xr.DataArray(sst_normalized,
                        dims=['n_model', 'n_mon', 'lat', 'lon'],
                        coords={'n_model': ds['n_model'], 'n_mon': ds['n_mon'], 'lat': ds['lat'], 'lon': ds['lon']},
                        name='sssNor')
        
    # 保存標準差作為附加變數
    std_da = xr.DataArray(std,
                        dims=['n_model', 'lat', 'lon'],
                        coords={'n_model': ds['n_model'], 'lat': ds['lat'], 'lon': ds['lon']},
                        name='stdsss')
        
    
    # 合併數據和標準差
    output_ds = xr.merge([new_ds, std_da])
    output_ds.to_netcdf(os.path.join(out_dir,output_filename))
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

def expansion(in_dir, input_filename, n_model_num, out_dir, output_filename):
    input_nc_file = os.path.join(in_dir, input_filename)
    ds = xr.open_dataset(input_nc_file)
    print(ds)

    # lon_name = input("print the longitude name:")
    # lat_name = input("print the latitude name:")

    # ds = ds.rename({'time': 'n_mon'})
    # print(ds)

    # expand dimensions
    sss_expanded = ds["sos"].expand_dims(dim={"n_model": [n_model_num]})  # shape: (n_model, time, lev, lat, lon)
    # sss_expanded = sss_expanded.assign_coords(n_mon=np.arange(ds.sizes['time']))
    sss_expanded = sss_expanded.transpose("n_model", "n_mon", "lat", "lon")

    # 組合新的 Dataset
    new_ds = xr.Dataset(
        {
            "sss": sss_expanded,
            # "sssy": sss_expanded,

        },
        coords={
            "n_model": [n_model_num],
            "n_mon": ds.n_mon,
            "lat": ds.lat,
            "lon": ds.lon
        }
    )

    print(new_ds)
    # 儲存
    output_file = os.path.join(out_dir, output_filename)
    new_ds.to_netcdf(output_file)



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
    ds = xr.open_dataset(sss_convert)
    print(ds)
    norm(sss_convert, 'sss', output_dir, 'sss_norm_1850_2024.nc', 'sssNor')
    
    # 6. 合併 SST 和 SSS 數據
    sss_norm_file = output_dir + "sss_norm_1850_2024.nc"
    merge_with_nino34(sss_norm_file, sss_norm_file, output_dir, 'GFDL-CM4_sss_sst_nino_1850_2024.nc')
    merge(sss_norm_file, sss_norm_file, output_dir, 'GFDL-CM4_sss_sst_1850_2024.nc')

    print("all_data process successful!")


# 執行流程
if __name__ == "__main__":
    cmip6_dir = './cmip6_sos_data/'
    inter_dir = './cmip6_sos_data/inter'
    norm_dir = './cmip6_sos_data/norm'

    ACCESS_CM2_dir = './cmip6_sos_data/ACCESS-CM2/'
    ACCESS_ESM1_5_dir = './cmip6_sos_data/ACCESS-ESM1-5/'
    CanESM5_CanOE_dir = './cmip6_sos_data/CanESM5-CanOE/'
    CMCC_CM2_HR4_dir = './cmip6_sos_data/CMCC-CM2-HR4/'
    CESM2_WACCM_FV2_dir = './cmip6_sos_data/CESM2-WACCM-FV2/'
    EC_Earth3_CC_dir = './cmip6_sos_data/EC-Earth3-CC/'
    FGOALS_f3_L_dir = './cmip6_sos_data/FGOALS-f3-L/'
    FGOALS_g3_dir = './cmip6_sos_data/FGOALS-g3/'
    GFDL_CM4_dir = './cmip6_sos_data/GFDL-CM4/'
    GFDL_ESM4_dir = './cmip6_sos_data/GFDL-ESM4/'
    GISS_E2_1_G_dir = './cmip6_sos_data/GISS-E2-1-G/'
    GISS_E2_1_H_dir = './cmip6_sos_data/GISS-E2-1-H/'
    MIROC6_dir = './cmip6_sos_data/MIROC6/'
    MPI_ESM_1_2_HAM_dir = './cmip6_sos_data/MPI-ESM-1-2-HAM/'
    MPI_ESM1_2_LR_dir = './cmip6_sos_data/MPI-ESM1-2-LR/'
    NESM3_dir = './cmip6_sos_data/NESM3/'
    NorESM2_MM_dir = './cmip6_sos_data/NorESM2-MM/'
    SAM0_UNICON_dir = './cmip6_sos_data/SAM0-UNICON/'

    ACCESS_CM2_inter_dir = './cmip6_sos_data/inter/ACCESS-CM2/'
    ACCESS_ESM1_5_inter_dir = './cmip6_sos_data/inter/ACCESS-ESM1-5/'
    CanESM5_CanOE_inter_dir = './cmip6_sos_data/inter/CanESM5-CanOE/'
    CMCC_CM2_HR4_inter_dir = './cmip6_sos_data/inter/CMCC-CM2-HR4/'
    CESM2_WACCM_FV2_inter_dir = './cmip6_sos_data/inter/CESM2-WACCM-FV2/'
    EC_Earth3_CC_inter_dir = './cmip6_sos_data/inter/EC-Earth3-CC/'
    FGOALS_f3_L_inter_dir = './cmip6_sos_data/inter/FGOALS-f3-L/'
    FGOALS_g3_inter_dir = './cmip6_sos_data/inter/FGOALS-g3/'
    GFDL_CM4_inter_dir = './cmip6_sos_data/inter/GFDL-CM4/'
    GFDL_ESM4_inter_dir = './cmip6_sos_data/inter/GFDL-ESM4/'
    GISS_E2_1_G_inter_dir = './cmip6_sos_data/inter/GISS-E2-1-G/'
    GISS_E2_1_H_inter_dir = './cmip6_sos_data/inter/GISS-E2-1-H/'
    MIROC6_inter_dir = './cmip6_sos_data/inter/MIROC6/'
    MPI_ESM_1_2_HAM_inter_dir = './cmip6_sos_data/inter/MPI-ESM-1-2-HAM/'
    MPI_ESM1_2_LR_inter_dir = './cmip6_sos_data/inter/MPI-ESM1-2-LR/'
    NESM3_inter_dir = './cmip6_sos_data/inter/NESM3/'
    NorESM2_MM_inter_dir = './cmip6_sos_data/inter/NorESM2-MM/'
    SAM0_UNICON_inter_dir = './cmip6_sos_data/inter/SAM0-UNICON/'

    # 1.將經度轉換成 0~360 度
    # convert_lon('cmip6_sos_data/ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc', cmip6_dir, 'ACCESS-CM2_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc', cmip6_dir, 'ACCESS-ESM1-5_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/CanESM5-CanOE_historical_r1i1p2f1_gn_185001-201412.nc', cmip6_dir, 'CanESM5-CanOE_185001-201412_convert.nc')
    
    # interpolate_to_regular_grid('cmip6_sos_data/CMCC-CM2-HR4_185001-201412.nc', CMCC_CM2_HR4_dir, output_prefix='sos',)
    # interpolate_to_regular_grid('cmip6_sos_data/EC-Earth3-CC_185001-201412.nc', EC_Earth3_CC_dir, output_prefix='sos',)
    # interpolate_to_regular_grid('cmip6_sos_data/FGOALS-f3-L_historical_r2i1p1f1_gn_185001-201412.nc', FGOALS_f3_L_dir, output_prefix='sos',)
    # interpolate_to_regular_grid('cmip6_sos_data/FGOALS-g3_185001-201412.nc', FGOALS_g3_dir, output_prefix='sos',)
    # interpolate_to_regular_grid('cmip6_sos_data/MPI-ESM-1-2-HAM_185001-201412.nc', MPI_ESM_1_2_HAM_dir, output_prefix='sos',)
    # interpolate_to_regular_grid('cmip6_sos_data/MPI-ESM1-2-LR_185001-201412.nc', MPI_ESM1_2_LR_dir, output_prefix='sos',)
    # interpolate_to_regular_grid('cmip6_sos_data/NESM3_historical_r1i1p1f1_gn_185001-201412.nc', NESM3_dir, output_prefix='sos',)
    # interpolate_to_regular_grid('cmip6_sos_data/NorESM2-MM_185001-201412.nc', NorESM2_MM_dir, output_prefix='sos',)
    # interpolate_to_regular_grid('cmip6_sos_data/SAM0-UNICON_185001-201412.nc', SAM0_UNICON_dir, output_prefix='sos',)
   
    # nc_combine(CMCC_CM2_HR4_dir, 'sos', cmip6_dir, 'CMCC-CM2-HR4_1850-2014_convert.nc')
    # nc_combine(EC_Earth3_CC_dir, 'sos', cmip6_dir, 'EC-Earth3-CC_1850-2014_convert.nc')
    # nc_combine(FGOALS_f3_L_dir, 'sos', cmip6_dir, 'FGOALS-f3-L_1850-2014_convert.nc')
    # nc_combine(FGOALS_g3_dir, 'sos', cmip6_dir, 'FGOALS-g3_1850-2014_convert.nc')
    # nc_combine(MPI_ESM_1_2_HAM_dir, 'sos', cmip6_dir, 'MPI-ESM-1-2-HAM_1850-2014_convert.nc')
    # nc_combine(MPI_ESM1_2_LR_dir, 'sos', cmip6_dir, 'MPI-ESM1-2-LR_1850-2014_convert.nc')
    # nc_combine(NESM3_dir, 'sos', cmip6_dir, 'NESM3_1850-2014_convert.nc')
    # nc_combine(NorESM2_MM_dir, 'sos', cmip6_dir, 'NorESM2-MM_1850-2014_convert.nc')
    # nc_combine(SAM0_UNICON_dir, 'sos', cmip6_dir, 'SAM0-UNICON_1850-2014_convert.nc')
   
    # convert_lon('cmip6_sos_data/CMCC-CM2-HR4_185001-201412.nc', cmip6_dir, 'CMCC-CM2-HR4_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/EC-Earth3-CC_185001-201412.nc', cmip6_dir, 'EC-Earth3-CC_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/FGOALS-f3-L_historical_r2i1p1f1_gn_185001-201412.nc', cmip6_dir, 'FGOALS-f3-L_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/FGOALS-g3_185001-201412.nc', cmip6_dir, 'FGOALS-g3_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/MPI-ESM-1-2-HAM_185001-201412.nc', cmip6_dir, 'MPI-ESM-1-2-HAM_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/MPI-ESM1-2-LR_185001-201412.nc', cmip6_dir, 'MPI-ESM1-2-LR_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/NESM3_historical_r1i1p1f1_gn_185001-201412.nc', cmip6_dir, 'NESM3_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/NorESM2-MM_185001-201412.nc', cmip6_dir, 'NorESM2-MM_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/SAM0-UNICON_185001-201412.nc', cmip6_dir, 'SAM0-UNICON_185001-201412_convert.nc')
    
    # convert_lon('cmip6_sos_data/AWI-CM-1-1-MR_185001-201412.nc', cmip6_dir, 'AWI-CM-1-1-MR_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/CESM2_185001_201412.nc', cmip6_dir, 'CESM2_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/CESM2-WACCM_historical_r1i1p1f1_gn_185001-201412.nc', cmip6_dir, 'CESM2-WACCM_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/CESM2-WACCM-FV2_185001-201412.nc', cmip6_dir, 'CESM2-WACCM-FV2_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/E3SM-1-1_185001-201412.nc', cmip6_dir, 'E3SM-1-1_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/GFDL-CM4_185001-201412.nc', cmip6_dir, 'GFDL-CM4_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/GFDL-ESM4_185001-201412.nc', cmip6_dir, 'GFDL-ESM4_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/GISS-E2-1-G_185001-201412.nc', cmip6_dir, 'GISS-E2-1-G_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/GISS-E2-1-H_185001-201412.nc', cmip6_dir, 'GISS-E2-1-H_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/IPSL-CM6A-LR_historical_r1i1p1f1_gn_185001-201412.nc', cmip6_dir, 'IPSL-CM6A-LR_185001-201412_convert.nc')
    # convert_lon('cmip6_sos_data/MIROC6_185001-201412.nc', cmip6_dir, 'MIROC6_185001-201412_convert.nc')


    # 2. 線性插值
    # interpolated('cmip6_sos_data/ACCESS-CM2_historical_r1i1p1f1_gn_185001-201412.nc', ACCESS_CM2_dir, 'ACCESS-CM2')
    # nc_combine(ACCESS_CM2_dir, 'sos', cmip6_dir, 'ACCESS-CM2_1850-2014_inter.nc')

    # interpolated('cmip6_sos_data/ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc', ACCESS_ESM1_5_dir, 'ACCESS-ESM1-5')
    # nc_combine(ACCESS_ESM1_5_dir, 'sos', cmip6_dir, 'ACCESS-ESM1-5_1850-2014_inter.nc')

    # interpolated('cmip6_sos_data/AWI-CM-1-1-MR_185001-201412.nc', inter_dir, 'AWI-CM-1-1-MR')
    # nc_combine(inter_dir, 'sos', cmip6_dir, 'AWI-CM-1-1-MR_1850-2014_inter.nc')

    # interpolated('cmip6_sos_data/CanESM5-CanOE_historical_r1i1p2f1_gn_185001-201412.nc', CanESM5_CanOE_dir, 'CanESM5-CanOE')
    # nc_combine(CanESM5_CanOE_dir, 'sos', cmip6_dir, 'CanESM5-CanOE_1850-2014_inter.nc')

    # interpolated('cmip6_sos_data/CMCC-CM2-HR4_1850-2014_convert.nc', CMCC_CM2_HR4_inter_dir, 'CMCC-CM2-HR4')
    # nc_combine(CMCC_CM2_HR4_inter_dir, 'sos', cmip6_dir, 'CMCC-CM2-HR4_1850-2014_inter.nc')
    
    # lat lon (1D)
    # interpolated('cmip6_sos_data/E3SM-1-1_185001-201412.nc', inter_dir, 'E3SM-1-1')
    # nc_combine(inter_dir, 'sos', cmip6_dir, 'E3SM-1-1_1850-2014_inter.nc')

    # interpolated('cmip6_sos_data/EC-Earth3-CC_1850-2014_convert.nc', EC_Earth3_CC_inter_dir, 'EC-Earth3-CC')
    # nc_combine(EC_Earth3_CC_inter_dir, 'sos', cmip6_dir, 'EC-Earth3-CC_1850-2014_inter.nc')

    # interpolated('cmip6_sos_data/FGOALS-f3-L_1850-2014_convert.nc', FGOALS_f3_L_inter_dir, 'FGOALS-f3-L')
    # nc_combine(FGOALS_f3_L_inter_dir, 'sos', cmip6_dir, 'FGOALS-f3-L_1850-2014_inter.nc')

    # interpolated('cmip6_sos_data/FGOALS-g3_1850-2014_convert.nc', FGOALS_g3_inter_dir, 'FGOALS-g3')
    # nc_combine(FGOALS_g3_inter_dir, 'sos', cmip6_dir, 'FGOALS-g3_1850-2014_inter.nc')

    # interpolated('cmip6_sos_data/MPI-ESM-1-2-HAM_1850-2014_convert.nc', MPI_ESM_1_2_HAM_inter_dir, 'MPI-ESM-1-2-HAM')
    # nc_combine(MPI_ESM_1_2_HAM_inter_dir, 'sos', cmip6_dir, 'MPI-ESM-1-2-HAM_1850-2014_inter.nc')

    # interpolated('cmip6_sos_data/MPI-ESM1-2-LR_1850-2014_convert.nc', MPI_ESM1_2_LR_inter_dir, 'MPI-ESM1-2-LR')
    # nc_combine(MPI_ESM1_2_LR_inter_dir, 'sos', cmip6_dir, 'MPI-ESM1-2-LR_1850-2014_inter.nc')

    # interpolated('cmip6_sos_data/NESM3_1850-2014_convert.nc', NESM3_inter_dir, 'NESM3')
    # nc_combine(NESM3_inter_dir, 'sos', cmip6_dir, 'NESM3_1850-2014_inter.nc')

    # interpolated('cmip6_sos_data/NorESM2-MM_1850-2014_convert.nc', NorESM2_MM_inter_dir, 'NorESM2-MM')
    # nc_combine(NorESM2_MM_inter_dir, 'sos', cmip6_dir, 'NorESM2-MM_1850-2014_inter.nc')

    # interpolated('cmip6_sos_data/SAM0-UNICON_1850-2014_convert.nc', SAM0_UNICON_inter_dir, 'SAM0-UNICON')
    # nc_combine(SAM0_UNICON_inter_dir, 'sos', cmip6_dir, 'SAM0-UNICON_1850-2014_inter.nc')

    # 3.計算異常值
    # da = xr.open_dataset('./cmip6_sos_data/ACCESS-CM2_1850-2014_inter.nc')
    # da = da.rename({'time': 'n_mon'})
    # da = da.assign_coords(n_mon=np.arange(da.sizes["n_mon"]))
    # da.to_netcdf(os.path.join(cmip6_dir, 'ACCESS-CM2_1850-2014_inter_1.nc'))  ##对拼接好的文件进行存储
    
    # da = xr.open_dataset('./cmip6_sos_data/ACCESS-ESM1-5_1850-2014_inter.nc')
    # da = da.rename({'time': 'n_mon'})
    # da = da.assign_coords(n_mon=np.arange(da.sizes["n_mon"]))
    # da.to_netcdf(os.path.join(cmip6_dir, 'ACCESS-ESM1-5_1850-2014_inter_1.nc')) 

    # da = xr.open_dataset('./cmip6_sos_data/CanESM5-CanOE_1850-2014_inter.nc')
    # da = da.rename({'time': 'n_mon'})
    # da = da.assign_coords(n_mon=np.arange(da.sizes["n_mon"]))
    # da.to_netcdf(os.path.join(cmip6_dir, 'CanESM5-CanOE_1850-2014_inter_1.nc')) 

    # anomaly('./cmip6_sos_data/ACCESS-CM2_1850-2014_inter_1.nc', cmip6_dir, 'ACCESS-CM2_anomaly.nc')    
    # anomaly('./cmip6_sos_data/ACCESS-ESM1-5_1850-2014_inter_1.nc', cmip6_dir, 'ACCESS-ESM1-5_anomaly.nc')
    # anomaly('./cmip6_sos_data/CanESM5-CanOE_1850-2014_inter_1.nc', cmip6_dir, 'CanESM5-CanOE_anomaly.nc')
    # anomaly('./cmip6_sos_data/CMCC-CM2-HR4_1850-2014_inter.nc', cmip6_dir, 'CMCC-CM2-HR4_anomaly.nc')
    # anomaly('./cmip6_sos_data/FGOALS-f3-L_1850-2014_inter.nc', cmip6_dir, 'FGOALS-f3-L_anomaly.nc')
    # anomaly('./cmip6_sos_data/FGOALS-g3_1850-2014_inter.nc', cmip6_dir, 'FGOALS-g3_anomaly.nc')
    # anomaly('./cmip6_sos_data/MPI-ESM-1-2-HAM_1850-2014_inter.nc', cmip6_dir, 'MPI-ESM-1-2-HAM_anomaly.nc')
    # anomaly('./cmip6_sos_data/MPI-ESM1-2-LR_1850-2014_inter.nc', cmip6_dir, 'MPI-ESM1-2-LR_anomaly.nc')
    # anomaly('./cmip6_sos_data/NESM3_1850-2014_inter.nc', cmip6_dir, 'NESM3_anomaly.nc')
    # anomaly('./cmip6_sos_data/NorESM2-MM_1850-2014_inter.nc', cmip6_dir, 'NorESM2-MM_anomaly.nc')
    
    # anomaly('./cmip6_sos_data/SAM0-UNICON_1850-2014_inter.nc', cmip6_dir, 'SAM0-UNICON_anomaly.nc')

    # 4.增加維度
    # da = xr.open_dataset('./cmip6_sos_data/CanESM5-CanOE_anomaly.nc')
    # da = da.rename({'j': 'lat', 'i':'lon'})
    # print(da)
    # da.to_netcdf(os.path.join(cmip6_dir, 'CanESM5-CanOE_anomaly_1.nc')) 
    # expansion(cmip6_dir, 'ACCESS-CM2_anomaly_1.nc', 0, cmip6_dir, 'ACCESS-CM2_anomaly_model_0.nc')
    # expansion(cmip6_dir, 'ACCESS-ESM1-5_anomaly_1.nc', 1, cmip6_dir, 'ACCESS-ESM1-5_anomaly_model_1.nc')
    # expansion(cmip6_dir, 'CanESM5-CanOE_anomaly_1.nc', 2, cmip6_dir, 'CanESM5-CanOE_anomaly_model_2.nc')
    # expansion(cmip6_dir, 'CMCC-CM2-HR4_anomaly.nc', 3, cmip6_dir, 'CMCC-CM2-HR4_anomaly_model_3.nc')
    # expansion(cmip6_dir, 'FGOALS-f3-L_anomaly.nc', 4, cmip6_dir, 'FGOALS-f3-L_anomaly_model_4.nc')
    # expansion(cmip6_dir, 'FGOALS-g3_anomaly.nc', 5, cmip6_dir, 'FGOALS-g3_anomaly_model_5.nc')
    # expansion(cmip6_dir, 'MPI-ESM-1-2-HAM_anomaly.nc', 6, cmip6_dir, 'MPI-ESM-1-2-HAM_anomaly_model_6.nc')
    # expansion(cmip6_dir, 'MPI-ESM1-2-LR_anomaly.nc', 7, cmip6_dir, 'MPI-ESM1-2-LR_anomaly_model_7.nc')
    # expansion(cmip6_dir, 'NESM3_anomaly.nc', 8, cmip6_dir, 'NESM3_anomaly_model_8.nc')
    # expansion(cmip6_dir, 'NorESM2-MM_anomaly.nc', 9, cmip6_dir, 'NorESM2-MM_anomaly_model_9.nc')
    # expansion(cmip6_dir, 'SAM0-UNICON_anomaly.nc', 10, cmip6_dir, 'SAM0-UNICON_anomaly_model_10.nc')

    # # 5.正規化
    # norm('./cmip6_sos_data/ACCESS-CM2_anomaly_model_0.nc', norm_dir, 'ACCESS-CM2_anomaly_norm_model_0.nc')
    # norm('./cmip6_sos_data/ACCESS-ESM1-5_anomaly_model_1.nc', norm_dir, 'ACCESS-ESM1-5_anomaly_norm_model_1.nc')
    # norm('./cmip6_sos_data/CanESM5-CanOE_anomaly_model_2.nc', norm_dir, 'CanESM5-CanOE_anomaly_norm_model_2.nc')
    # norm('./cmip6_sos_data/CMCC-CM2-HR4_anomaly_model_3.nc', norm_dir, 'CMCC-CM2-HR4_anomaly_norm_model_3.nc')
    # norm('./cmip6_sos_data/FGOALS-f3-L_anomaly_model_4.nc', norm_dir, 'FGOALS-f3-L_anomaly_norm_model_4.nc')
    # norm('./cmip6_sos_data/FGOALS-g3_anomaly_model_5.nc', norm_dir, 'FGOALS-g3_anomaly_norm_model_5.nc')
    # norm('./cmip6_sos_data/MPI-ESM-1-2-HAM_anomaly_model_6.nc', norm_dir, 'MPI-ESM-1-2-HAM_anomaly_norm_model_6.nc')
    # norm('./cmip6_sos_data/MPI-ESM1-2-LR_anomaly_model_7.nc', norm_dir, 'MPI-ESM1-2-LR_anomaly_norm_model_7.nc')
    # norm('./cmip6_sos_data/NESM3_anomaly_model_8.nc', norm_dir, 'NESM3_anomaly_norm_model_8.nc')
    # norm('./cmip6_sos_data/NorESM2-MM_anomaly_model_9.nc', norm_dir, 'NorESM2-MM_anomaly_norm_model_9.nc')
    # norm('./cmip6_sos_data/SAM0-UNICON_anomaly_model_10.nc', norm_dir, 'SAM0-UNICON_anomaly_norm_model_10.nc')

    # 6.結合所有model
    # model_combine(norm_dir, norm_dir, 'CMIP6_11_model_sss.nc')
    ds = xr.open_dataset('./cmip6_sos_data/CMCC-CM2-HR4_anomaly.nc', engine='netcdf4', decode_cf=False)
    # ds = xr.open_dataset('./cmip6_sos_data/norm/CMIP6_11_model_sss.nc', engine='netcdf4', decode_cf=False)
    print(ds)
    # print(ds['sos'].isel(time=0))
    ds = ds.isel(n_mon=0)
    # sos_approx = ds['sssNor'].values
    sos_approx = ds['sos'].values
    print(sos_approx)
    print(ds['lon'])

    lon = ds['lon']  # (i,)
    lat = ds['lat']   # (j,)
    # sos = ds['sssNor']        # (j, i)
    sos = ds['sos']        # (j, i)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(lon, lat, sos, shading='auto')
    plt.colorbar(label='Sea Surface Salinity (0.001 psu)')
    plt.title('Sea Surface Salinity')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    # ds['sos'].isel(time=0).plot()
    # plot_dataset(ds)

