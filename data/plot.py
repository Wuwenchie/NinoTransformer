import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np

def plot(dir, file_name):
    ds = xr.open_dataset(os.path.join(dir, file_name), engine='netcdf4', decode_cf=False)
    print(ds)

    # 提取經緯度
    # lat = ds['latitude']
    # lon = ds['longitude']
    sos = ds['sos'][0,:,:].values
    print('sos 原始範圍:', sos.min(), sos.max())
    # print(ds['sos'].attrs)

    # # 應用縮放（如果存在）
    # if 'scale_factor' in ds['sos'].attrs and 'add_offset' in ds['sos'].attrs:
    #     sos = sos * ds['sos'].scale_factor + ds['sos'].add_offset

    # # 過濾無效值
    # fill_value = ds['sos'].attrs.get('_FillValue', 1e+20)
    # sos = np.where(sos < 0.9 * fill_value, sos, np.nan)
    # print('過濾後 sos 範圍:', np.nanmin(sos), np.nanmax(sos))
        
    # # sos = np.where(sos < 1e+10, sos, np.nan)
    # # print('sos 範圍:', sos.min(), sos.max())

    # # 轉換經度範圍（如果需要）
    # lon = xr.where(lon > 180, lon - 360, lon)

    # # 計算 j 索引範圍（緯度 20°S 到 20°N）
    # j_mask = (lat >= -20) & (lat <= 20)
    # j_indices = np.where(j_mask.any(dim='i'))[0]
    # j_start = j_indices.min()
    # j_end = j_indices.max() + 1
    # print("j 索引範圍:", j_start, j_end)

    # # 計算 i 索引範圍（經度 120°E 到 90°W）
    # i_mask = (lon >= -90) & (lon <= 120)
    # i_indices = np.where(i_mask.any(dim='j'))[0]
    # i_start = i_indices.min()
    # i_end = i_indices.max() + 1
    # print("i 索引範圍:", i_start, i_end)

    # # 選擇指定範圍的 sos 數據
    # sos = ds['sos'].isel(time=-1, j=slice(j_start, j_end), i=slice(i_start, i_end))
    
    
    sos = ds['sos'].isel(n_mon=0)  # 讀取 sos 變量

    # # 關閉所有圖形並創建新圖形
    plt.close('all')
    fig = plt.figure(figsize=(12, 6))  # 設置圖形大小，1400/100=14 英寸，800/100=8 英寸
    # # plt.gcf().set_position([100, 50, 1400, 800])  # 設置窗口位置和大小（單位為像素）

    # # 旋轉 270 度並水平翻轉（等效於 rot90(sos, 3) 後 fliplr）
    # # sos_rotated = np.fliplr(np.rot90(sos, 3))

    # # 繪製 2D 偽顏色圖
    p = plt.contourf(sos, levels=np.arange(28, 38, 0.5), cmap='viridis')

    # # 設置顏色範圍
    plt.clim(28, 38)

    # 定義顏色條參數
    cbar_kwargs = {
        'orientation': 'horizontal',
        'shrink': 0.6,
        'pad': 0.05,
        'aspect': 40,
        'label': 'Sea Surface Salinity (PSU)'
    }

    # 添加顏色條並應用 cbar_kwargs
    plt.colorbar(p, **cbar_kwargs)
    # 添加顏色條
    # plt.colorbar(p)
    plt.title('ACCESS-CM2 reanalysis Sea Surface Salinity after interploted - Jan 1850 (range=180°E~180°W, 20°S~20°N)')

    # plt.savefig('./cmip6_sos_data/ACCESS-CM2/access-cm2_interploted_185001')
    # 顯示圖形
    plt.show()

plot('./cmip6_sos_data', 'CMCC-CM2-HR4_1850-2014_inter.nc')
