import xarray as xr
from typing import List
from warnings import warn
import numpy as np
import traceback
import cmath
import pandas as pd
from scipy.ndimage import label, generate_binary_structure
from skimage.measure import regionprops_table
import bottleneck
from scipy import stats
import requests
from subprocess import call
import os
import warnings
from urllib.parse import urlparse
import xesmf as xe
from warnings import warn


def full_stack_error():
    import traceback, sys
    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]       # remove call of full_stack, the printed exception
                            # will contain the caught exception caller instead
    trc = 'Traceback (most recent call last):\n'
    stackstr = trc + ''.join(traceback.format_list(stack))
    if exc is not None:
         stackstr += '  ' + traceback.format_exc().lstrip(trc)
    return stackstr


def regrid_ds1_on_ds2(ds1, ds2, method='conservative',
                 latname='latitude', lonname='longitude', copy=True):
    if copy:
        ds1 = ds1.copy()
        ds2 = ds2.copy()
    if ds1.name is None:
        ds1.name = 'ds1'
    if ds2.name is None:
        ds2.name = 'ds2'

    ds1 = ds1.sortby(lonname).sortby(latname)
    ds2 = ds2.sortby(lonname).sortby(latname)
    lon0_b = ds2[lonname].values[0]
    lon1_b = ds2[lonname].values[-1]
    dlon = ds2[lonname].diff(lonname).values[0]
    lat0_b = ds2[latname].values[0]
    lat1_b = ds2[latname].values[-1]
    dlat = ds2[latname].diff(latname).values[0]
    ds_out = xe.util.grid_2d(lon0_b, lon1_b, dlon, lat0_b, lat1_b, dlat)
    if not isinstance(ds1, xr.Dataset):
        is_array = True
        ds1 = ds1.to_dataset()
    else:
        is_array = False

    regridder = xe.Regridder(ds1, ds_out, method=method)
    ds_regridded = regridder(ds1)
    ds_regridded = ds_regridded.assign_coords(x=ds_out['lon'].values[0, :],
                                       y=ds_out['lat'].values[:, 0]).rename(x=lonname, y=latname)
    if is_array:
        ds_regridded = ds_regridded.to_array().isel(variable=0).drop('variable')

    try:
        xr.testing.assert_allclose(ds_regridded[latname], ds2[latname])
        xr.testing.assert_allclose(ds_regridded[lonname], ds2[lonname])
    except AssertionError:
        warn('Regridded coords are not the same, probably because ds2 boundaries are outside ds2. \n'
             'This method does not extrapolate. I will reindex instead.')
        ds_regridded = ds_regridded.reindex({latname: ds2[latname],
                                             lonname: ds2[lonname]}, method='nearest')
    if latname != 'lat':
        ds_regridded = ds_regridded.drop('lat')
    if lonname != 'lon':
        ds_regridded = ds_regridded.drop('lon')

    return ds_regridded


def apply_regrid(ds, method='conservative', grid_spacing=[5, 5],
                 latname='latitude', lonname='longitude', copy=True):
    if copy:
        ds = ds.copy()
    if ds.name is None:
        ds.name = 'placeholder_name'
    ds = ds.sortby(lonname).sortby(latname)
    lon0_b = ds[lonname].values[0]
    lon1_b = ds[lonname].values[-1]
    dlon = grid_spacing[0]
    lat0_b = ds[latname].values[0]
    lat1_b = ds[latname].values[-1]
    dlat = grid_spacing[1]
    ds_out = xe.util.grid_2d(lon0_b, lon1_b, dlon, lat0_b, lat1_b, dlat)
    if not isinstance(ds, xr.Dataset):
        is_array = True
        ds = ds.to_dataset()
    else:
        is_array = False

    regridder = xe.Regridder(ds, ds_out, method=method)
    ds_regridded = regridder(ds)
    ds_regridded = ds_regridded.assign_coords(x=ds_out['lon'].values[0, :],
                                       y=ds_out['lat'].values[:, 0]).rename(x=lonname, y=latname)
    if is_array:
        ds_regridded = ds_regridded.to_array().isel(variable=0).drop('variable')
    return ds_regridded


def covariance_gufunc(x, y):
    return ((x - x.mean(axis=-1, keepdims=True))
            * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)


def pearson_correlation_gufunc(x, y):
    return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))


def spearman_correlation_gufunc(x, y):
    x_ranks = bottleneck.rankdata(x, axis=-1)
    y_ranks = bottleneck.rankdata(y, axis=-1)
    return pearson_correlation_gufunc(x_ranks, y_ranks)


def spearman_correlation(x, y, dim, outdtype=float):
    return xr.apply_ufunc(
        spearman_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask='allowed',
        output_dtypes=[outdtype])


def spearman_pvalue(x, y, dim):
    return xr.apply_ufunc(lambda a, b: stats.spearmanr(a, b)[1], x, y,
                          input_core_dims=[[dim], [dim]],
                          dask='allowed',
                          output_dtypes=[float])


def size_in_memory(da):
    """
    Check xarray (dask or not) size in memory without loading
    Parameters
    ----------
    da
    -------

    """
    if da.dtype == 'float64':
        size = 64
    elif da.dtype == 'float32':
        size = 32
    else:
        raise TypeError('array dtype not recognized.')

    n_positions = np.prod(da.shape)
    total_size = size * n_positions / (8 * 1e9)
    print(f'Total array size is: {total_size} GB')
    return None


def filter_ridges(ridges, ftle, criteria, thresholds, verbose=True):
    """
    Method to filter 2 or 3d arrays based on pandas regionprops
    Parameters
    ----------
    ridges xr.Dataarray boolean
    ftle xr.Dataarray floats
    criteria list of strings with regionprops criteria
    thresholds list of thresholds for the criteria in the same order

    Returns
    -------
    Filtered xr.Dataarray
    """
    verboseprint = print if verbose else lambda *a, **k: None


    # --- Asserting dimensions --- #
    dims = ridges.dims
    dims_ftle = ftle.dims
    assert set(dims) == set(dims_ftle), 'Dims must be equal'
    not_time_dims = set(dims).difference({'time'})
    ftle = ftle.transpose(..., *not_time_dims)
    ridges = ridges.transpose(..., *not_time_dims)

    # --- Labelling features --- #
    s = generate_binary_structure(2, 2)  # Full connectivity in space
    if len(dims) == 3:
        s = np.stack([np.zeros_like(s), s, np.zeros_like(s)])  # No connectivity in time
    ridges_labels = ridges.copy(data=label(ridges, structure=s)[0])

    # --- Measuring properties --- #

    verboseprint('#----- Applying regionprops -----#')
    props = ['label'] + criteria
    if len(dims) < 3:
        df = pd.DataFrame(regionprops_table(ridges_labels.values,
                                            intensity_image=ftle.values,
                                            properties=props))
    else:
        df_list = []
        for idx, time in enumerate(ridges_labels.time.values):
            rl = ridges_labels.sel(time=time)
            df = pd.DataFrame(regionprops_table(rl.values,intensity_image=ftle.sel(time=time).values,properties=props))
            verboseprint(f'done time {idx} of {ridges_labels.time.values.shape[0]}')
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
    df.set_index('label', inplace=True)

    # --- Creating masks --- #

    verboseprint('#----- Creating masks -----#')
    masks = []
    for idx, criterion in enumerate(criteria):
        da = df.to_xarray()[criterion]
        da = da.where(da > thresholds[idx], drop=True)
        mask = ridges_labels.copy()
        mask = mask.isin(da.label.values.tolist())
        # for l in da.label.values:
        #     mask = mask.where(mask != l, 1)
        mask = mask.where(mask == 1, 0)
        masks.append(mask)

    # --- Applying masks --- #
    verboseprint('#----- Applying masks -----#')

    mask_final = masks[0]
    if len(masks) > 1:
        for mask in masks[1:]:
            mask_final = mask_final * mask
    ridges_labels = mask_final.where(mask_final)

    ridges_labels = ridges_labels.transpose(*dims)  # Returning original order
    ftle = ftle.transpose(*dims)  # Returning to original order since it is not a deepcopy
    return ridges_labels


def common_index(list1, list2):
    return [element for element in list1 if element in list2]


def calc_quantiles(x, quantiles_list, iterations=50):
    quantile_array = np.zeros(shape=[len(quantiles_list), iterations])
    for i in range(iterations):
        for j, q in enumerate(quantiles_list):
            quantile_array[j, i] = np.quantile(x, q)

    return np.mean(quantile_array, axis=1), np.std(quantile_array, axis=1)

def exists(URL):
    r = requests.head(URL)
    return r.status_code == requests.codes.ok



def is_url(url):
  try:
    result = urlparse(url)
    return all([result.scheme, result.netloc])
  except ValueError:
    return False

def download_and_sanitize(URL=None):

    bom =True
    if bom:
        if URL is None:
            # URL = 'http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt'  # BOM Website
            URL = 'https://meteorologia.unifei.edu.br/teleconexoes/cache/mjo.txt'  # BOM Website
        columns_to_select = ['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude']
        assert is_url(URL), 'Path does not exist'
        # call(["wget", "--user-agent=Mozilla", "-O", 'temp.txt', URL], stdout=open(os.devnull, 'wb'))
        call(["curl", "-k", "-s", "-o", 'temp.txt', URL], stdout=open(os.devnull, 'wb'))
        df = pd.read_csv('temp.txt', skiprows=1, delim_whitespace=True) # REad bom

        call(['rm', 'temp.txt'])
        new_columns = []
        for idx, column in enumerate(df.columns):
            new_columns.append(column.replace(',', '').replace('.', ''))
        df.columns = pd.Index(new_columns)
        df = df[columns_to_select]
        df['time'] = pd.to_datetime(df[['year', 'month', 'day']])
        df = df.drop(['year', 'month', 'day'], axis=1)
        df = df.set_index('time')

    else:
        URL = 'https://psl.noaa.gov/mjo/mjoindex/vpm.1x.txt'  # NOAA
        assert is_url(URL), 'Path does not exist'
        call(["curl", "-s", "-o", 'temp.txt', URL], stdout=open(os.devnull, 'wb'))
        df = pd.read_csv('temp.txt', header=None, delim_whitespace=True)  # REad bom
        df.columns =['year', 'month', 'day','hour', 'RMM1', 'RMM2','amplitude']
        df['time'] = pd.to_datetime(df[['year', 'month', 'day','hour']])

        df = df.set_index('time')
        df = df.loc[:,['RMM1','RMM2']]
        call(['rm', 'temp.txt'])

    ds = df.to_xarray()
    ds = ds.where(ds < 1e10)
    ds = ds.where(ds != 999)
    return ds


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


def safely_read_multiple_files_dask(files, size_of_chunk_in=20, size_of_chunk_out=20, concat_dim='time',
                                    concat_trajs=False, convert_time=False):
    """
    Method to read multiple netcdfs and concat along one dimension
    Parameters
    ----------
    files List of paths
    size_of_chunk_out Integer - number of files to concatenate at a time
    concat_dim String

    Returns concatenated xr.dataarray
    -------

    """

    import gc
    import pdb
    import threading
    try:
        import psutil
    except:
        psutil_available = False
    else:
        psutil_available = True
    def convert_date(x):
        return pd.Timestamp(str(x))
    print(('Number of active threads: ' + str(threading.active_count())))
    if psutil_available:
        p = psutil.Process()
        print('Process using ' + str(round(p.memory_percent(), 2)) + '% of the total system memory.')

    array_list = []
    chunked_array_list = []
    for i, file in enumerate(files):
        print('Reading file ' + str(i))
        try:
            da_temp = xr.open_dataarray(file, chunks={'time': size_of_chunk_in})
            if concat_trajs:
                warnings.warn('For now assuming that files that have time dimension are backtrajectories. \n'
                              'This needs to be fixed for a more general usage') # TODO
                da_temp = da_temp.rename({concat_dim: 'trajectory_time'})
                da_temp = da_temp.assign_coords({concat_dim: da_temp.trajectory_time.values[-1]})
                da_temp = da_temp.expand_dims(concat_dim)
                da_temp = da_temp.assign_coords(trajectory_time=np.arange(0, da_temp.trajectory_time.values.shape[0]))
            if convert_time:
                da_temp = da_temp.assign_coords(time=[convert_date(x) for x in da_temp.time.values])
            array_list.append(da_temp)


        except (OSError, ValueError):
            print('Reading file ' + str(i) + ' failed')
        else:
            if (i % size_of_chunk_out == 0) or (i == (len(files)-1)):
                if psutil_available:
                    print('Process using ' + str(round(p.memory_percent(), 2)) + '% of the total system memory.')
                print(('Number of active threads: ' + str(threading.active_count())))

                chunked_array_list.append(
                    xr.concat(array_list, dim=concat_dim, coords='minimal', compat='override')
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
    elif region == 'SA':
        domain = dict(latitude=[-45, 25], longitude=[-90, -20])

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
            x = x.assign_coords(**{lonName: (x[lonName].values + 180) % 360 - 180})
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
