import xarray as xr
from typing import List
from warnings import warn
import numpy as np
import traceback
import cmath
import pandas as pd
from skimage.feature import hessian_matrix_eigvals
from scipy.ndimage import gaussian_filter


def find_ridges_spherical_hessian(da, sigma=.5, scheme='first_order'):
    """
    Method to in spherical coordinates
    Parameters
    ----------
    sigma - float, smoothing intensity
    da - xarray.dataarray
    scheme - str, 'first_order' for x[i+1] - x[i] and second order for x[i+1] - x[i-1]

    Returns
    -------

    """
    da = da.copy()
    # Gaussian filter
    da = da.copy(data=gaussian_filter(da, sigma=sigma))

    # Initializing
    earth_r = 6371000
    x = da.longitude.copy() * np.pi/180
    y = da.latitude.copy() * np.pi/180
    dx = x.diff('longitude') * earth_r * np.cos(y)
    dy = y.diff('latitude') * earth_r
    dx_scaling = da.longitude.diff('longitude').values[0]  # grid spacing
    dy_scaling = da.latitude.diff('latitude').values[0]  # grid spacing

    # Calc derivatives
    if scheme == 'second_order':
        ddadx = dx_scaling * da.differentiate('longitude') / dx
        ddady = dy_scaling * da.differentiate('latitude') / dy
        d2dadx2 = dx_scaling * ddadx.differentiate('longitude') / dx
        d2dadxdy = dy_scaling * ddadx.differentiate('latitude') / dy
        d2dady2 = dx_scaling * ddady.differentiate('latitude') / dy
        d2dadydx = d2dadxdy.copy()
    elif scheme == 'first_order':
        ddadx = da.diff('longitude') / dx
        ddady = da.diff('latitude') / dy
        d2dadx2 = ddadx.diff('longitude') / dx
        d2dadxdy = ddadx.diff('latitude') / dy
        d2dady2 = ddady.diff('latitude') / dy
        d2dadydx = d2dadxdy.copy()
    # Assembling Hessian array
    hessian = xr.concat([d2dadx2, d2dadxdy, d2dadydx, d2dady2],
                            dim=pd.Index(['d2dadx2', 'd2dadxdy', 'd2dadydx', 'd2dady2'],
                                         name='elements'))
    hessian = hessian.stack({'points': ['latitude', 'longitude']})
    hessian = hessian.dropna('points', how='any')

    # Finding norm
    norm = hessian_matrix_eigvals([hessian.sel(elements='d2dadx2').values,
                                   hessian.sel(elements='d2dadxdy').values,
                                   hessian.sel(elements='d2dady2').values])
    norm_max = hessian.isel(elements=0).drop('elements').copy(data=norm[1, :]).unstack()

    return norm_max


def common_index(list1, list2):
    return [element for element in list1 if element in list2]


def calc_quantiles(x, quantiles_list, iterations=50):
    quantile_array = np.zeros(shape=[len(quantiles_list), iterations])
    for i in range(iterations):
        for j, q in enumerate(quantiles_list):
            quantile_array[j, i] = np.quantile(x, q)

    return np.mean(quantile_array, axis=1), np.std(quantile_array, axis=1)


def safely_read_multiple_files(files, size_of_chunk=20, concat_dim = 'time'):
    """
    Method to read multiple netcdfs and concat along one dimension
    Parameters
    ----------
    files List of paths
    size_of_chunk Integer - number of files to concatenate at a time
    concat_dim String

    Returns concatenated xr.dataarray
    -------

    """

    import gc
    import threading
    try:
        import psutil
    except:
        psutil_available = False
    else:
        psutil_available = True


    print(('Number of active threads: ' + str(threading.active_count())))
    if psutil_available:
        p = psutil.Process()
        print('Process using ' + str(round(p.memory_percent(), 2)) + '% of the total system memory.')

    array_list = []
    chunked_array_list = []
    for i, file in enumerate(files):
        print('Reading file ' + str(i))
        array_list.append(xr.open_dataarray(file))

        if i % size_of_chunk == 0:
            if psutil_available:
                print('Process using ' + str(round(p.memory_percent(), 2)) + '% of the total system memory.')
            print(('Number of active threads: ' + str(threading.active_count())))
            chunked_array_list.append(
                xr.concat(array_list, dim=concat_dim)
            )
            [file.close() for file in array_list]  # closing all handles
            array_list = []  #  resetting array
            gc.collect()

    return xr.concat(chunked_array_list, dim=concat_dim)


def latlonsel(array, latitude, longitude, latname='latitude', lonname='longitude'):
    """
    Function to crop array based on lat and lon intervals given by slice or list.
    This function is able to crop across cyclic boundaries.

    :param array: xarray.Datarray
    :param lat: list or slice (min, max)
    :param lon: list or slice(min, max)
    :return: cropped array
    """
    assert latname in array.coords, f"Coord. {latname} not present in array"
    assert lonname in array.coords, f"Coord. {lonname} not present in array"


    if isinstance(latitude, slice):
        lat1 = latitude.start
        lat2 = latitude.stop
    elif isinstance(latitude, list):
        lat1 = latitude[0]
        lat2 = latitude[-1]
    if isinstance(longitude, slice):
        lon1 = longitude.start
        lon2 = longitude.stop
    elif isinstance(longitude, list):
        lon1 = longitude[0]
        lon2 = longitude[-1]

    lonmask = (array[lonname] < lon2) & (array[lonname] > lon1)
    latmask = (array[latname] < lat2) & (array[latname] > lat1)
    array = array.where(lonmask, drop=True).where(latmask, drop=True)
    return array



def add_basin_coord(array, MAG):
    basin_names = list(MAG.coords.keys())
    basin_avg = {}
    for basin in basin_names:
        array.coords[basin] = (("latitude", "longitude"), MAG.coords[basin].values)

    return array


def get_xr_seq(ds, commun_sample_dim, idx_seq):
    """
    Internal function that create the sequence dimension
    :param ds:
    :param commun_sample_dim:
    :param idx_seq:
    :return:
    """
    dss = []
    for idx in idx_seq:
        dss.append(ds.shift({commun_sample_dim: -idx}))

    dss = xr.concat(dss, dim='seq')
    dss = dss.assign_coords(seq=idx_seq)

    return dss


def createDomains(region, reverseLat=False):
    if region == "SACZ_big":
        domain = dict(latitude=[-40, 5], longitude=[-70, -20])
    elif region == "SACZ":
        domain = dict(latitude=[-40, -5], longitude=[-62, -20])
    elif region == "SACZ_small":
        domain = dict(latitude=[-30, -20], longitude=[-50, -35])
    elif region == "AITCZ":
        domain = dict(latitude=[-5, 15], longitude=[-45, -1])
    elif region == "NEBR":
        # domain = dict(latitude=[-15, 5], longitude=[-45, -15])
        domain = dict(latitude=[-10, 5], longitude=[-55, -40])
    elif region is None:
        domain = dict(latitude=[None, None], longitude=[None, None])
    else:
        raise ValueError(f'Region {region} not supported')

    if reverseLat:
        domain = dict(latitude=slice(domain['latitude'][1], domain['latitude'][0]),
                      longitude=slice(domain['longitude'][0], domain['longitude'][1]))

    else:
        domain = dict(latitude=slice(domain['latitude'][0], domain['latitude'][1]),
                      longitude=slice(domain['longitude'][0], domain['longitude'][1]))

    return domain


def read_nc_files(region=None,
                  basepath="/group_workspaces/jasmin4/upscale/gmpp/convzones/",
                  filename="SL_repelling_{year}_lcstimelen_1.nc",
                  year_range=range(2000, 2008), transformLon=False, lonName="longitude", reverseLat=False, reverseLon=False,
                  time_slice_for_each_year=slice(None, None), season=None, lcstimelen=None, set_date=False,
                  binary_mask=None, maskdims={'latitude': 'lat', 'longitude': 'lon'}):
    """

    :param binary_mask:
    :param transformLon:
    :param lonName:
    :param region:
    :param basepath:
    :param filename:
    :param year_range:
    :return:
    """
    print("*---- Starting reading data ----*")
    years = year_range
    file_list = []

    def transform(x, binary_mask):
        if transformLon:
            x.coords[lonName].values = \
                (x.coords[lonName].values + 180) % 360 - 180
        if not isinstance(region, (type(None), dict)):
            x = x.sel(createDomains(region, reverseLat))
        elif isinstance(region, type(dict)):
            x = x.sel(region)
        if not isinstance(season, type(None)):
            if set_date:
                initial_date = pd.Timestamp(f'{year}-01-01T00:00:00') + pd.Timedelta(str(lcstimelen*6 - 6) + 'H')
                final_date = pd.Timestamp(f'{year}-12-31T18:00:00')
                freq = '6H'
                x = x.assign_coords(time=pd.date_range(initial_date, final_date, freq=freq))
            if season == 'DJF':
                season_idxs = np.array([pd.to_datetime(t).month in [1, 2, 12] for t in x.time.values])
            elif season == 'JJA':
                season_idxs = np.array([pd.to_datetime(t).month in [5, 6, 7] for t in x.time.values])
            else:
                raise ValueError(f"Season {season} not supported")
            x = x.sel(time=x.time[season_idxs])
        if isinstance(binary_mask, xr.DataArray):
            binary_mask = binary_mask.where(binary_mask == 1, drop=True)
            x = x.sel(latitude=binary_mask[maskdims['latitude']].values,
                      longitude=binary_mask[maskdims['longitude']].values, method='nearest')
            binary_mask = binary_mask.rename({maskdims['longitude']: 'longitude', maskdims['latitude']: 'latitude'})
            x = x.where(binary_mask == 1, drop=True)

        return x

    for i, year in enumerate(years):
        print(f'Reading year {year}')
        filename_formatted = basepath + filename.format(year=year)
        print(filename_formatted)
        year = str(year)
        array = None
        fs = (xr.open_dataarray, xr.open_dataset)
        for f in fs:
            try:
                array = f(filename_formatted)
            except ValueError:
                print('Could not open file using {}'.format(f.__name__))
            else:
                break

        if isinstance(array, (xr.DataArray, xr.Dataset)):
            file_list.append(transform(array, binary_mask).isel(time=time_slice_for_each_year))
        else:
            print(f'Year {year} unavailable')
    print(file_list)
    full_array = xr.concat(file_list, dim='time')
    print('*---- Finished reading data ----*')
    return full_array


# def get_xr_seq(ds: xr.DataArray, seq_dim: str, idx_seq: List[int]):
#    """
#    Function that create the sequence dimension in overlapping time intervals
#
#    :param ds:
#    :param seq_dim:
#    :param idx_seq:
#    :return: xr.DataArray
#    """
#    dss = []
#    for idx in idx_seq:
#        dss.append(ds.shift({seq_dim: -idx}))

#    dss = xr.concat(dss, dim='seq')
#    dss = dss.assign_coords(seq=idx_seq)

#    return dss


def get_seq_mask(ds: xr.DataArray, seq_dim: str, seq_len: int):
    """
    Function that create the sequence dimension in non-overlapping time intervals

    :param ds:
    :param seq_dim:
    :param idx_seq:
    :return: xr.DataArray
    """
    mask = []
    quotient, remainder = divmod(len(ds[seq_dim].values), seq_len)
    assert quotient > 0, f'seq_len cannot be larger than {seq_dim} length!'

    if remainder != 0:
        warn(f"Length of dim {seq_dim} is not divisible by seq_len {seq_len}. Dropping last {remainder} entries.")
        ds = ds.isel({seq_dim: slice(None, len(ds[seq_dim].values) - remainder)})

    print(ds[seq_dim])
    for i, time in enumerate(ds[seq_dim].values.tolist()):
        idx = int(i / seq_len)
        mask.append(idx)

    ds['seq'] = ((seq_dim), mask)
    return ds


def to_cartesian(array, lon_name='longitude', lat_name='latitude', earth_r=6371000):
    """
    Method to include cartesian coordinates in a lat lon xr. DataArray

    :param array: input xr.DataArray
    :param lon_name: name of the longitude dimension in the array
    :param lat_name: name of the latitude dimension in the array
    :param earth_r: earth radius
    :return: xr.DataArray with x and y cartesian coordinates
    """
    array['x'] = array[lon_name] * np.pi * earth_r / 180
    array['y'] = xr.apply_ufunc(lambda x: np.sin(np.pi * x / 180) * earth_r, array[lat_name])
    return array


def xy_to_latlon(x, y, earth_r=6371000):
    """
    Inverse function of meteomath.to_cartesian
    """
    longitude = x * 180 / (np.pi * earth_r)
    latitude = np.arcsin(y / earth_r) * 180 / np.pi
    return latitude, longitude


def get_xr_seq(ds, commun_sample_dim, idx_seq):
    """
    Internal function that create the sequence dimension
    :param ds:
    :param commun_sample_dim:
    :param idx_seq:
    :return:
    """
    dss = []
    for idx in idx_seq:
        dss.append(ds.shift({commun_sample_dim: -idx}))

    dss = xr.concat(dss, dim='seq')
    dss = dss.assign_coords(seq=idx_seq)

    return dss


def get_xr_seq_coords_only(ds, commun_sample_dim, idx_seq):
    """
    Internal function that create the sequence dimension
    :param ds:
    :param commun_sample_dim:
    :param idx_seq:
    :return:
    """
    dss = []
    for idx in idx_seq:
        dss.append(ds[commun_sample_dim].shift({commun_sample_dim: -idx}))

    dss = xr.concat(dss, dim='seq')
    dss = dss.assign_coords(seq=idx_seq)

    return dss
