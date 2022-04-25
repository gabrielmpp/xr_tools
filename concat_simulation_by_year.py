"""
Script to merge FTLE simulations
"""
import glob
import os
from xr_tools.tools import safely_read_multiple_files_dask
import logging
import dateutil.parser as dparser
from joblib import Parallel, delayed
import sys
from dask.diagnostics import ProgressBar
import pandas as pd
import numpy as np


def scan_folder(parent):
    # iterate over all the files in directory 'parent'
    found_file=0
    for file_name in os.listdir(parent):
        if file_name.endswith(".nc"):
            # if it's a txt file, print its name (or do whatever you want)
            found_file = 1
        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recursively call this method
                scan_folder(current_path)
    if found_file == 1:
        dirs_with_nc.append(parent)


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


def merge_files(dir_sim):
    try:
        os.makedirs(dir_sim + '/final/')
    except FileExistsError:
        os.rmdir(dir_sim + '/final/')
        os.makedirs(dir_sim + '/final/')
    all_files = glob.glob(dir_sim + '/*.nc')
    all_files.sort()

    # expected filename f1950-01-01T00:00:00.000000000.nc
    starting_year = dparser.parse(all_files[0].split('/')[-1].replace('f', '').split('T')[0], fuzzy=True).year
    final_year = dparser.parse(all_files[-1].split('/')[-1].replace('f', '').split('T')[0], fuzzy=True).year

    assert starting_year < final_year, 'Starting year larger than final year.'
    assert 1900 < starting_year < final_year < 2200, 'Years not in expected interval'
    for year in range(starting_year, final_year + 1):
        logging.info(f'\n RUNNING YEAR {str(year)} \n ')
        try:

            files_of_year = [f for f in all_files if str(year) in f]
            ds_year = safely_read_multiple_files_dask(files_of_year, size_of_chunk_in=1, size_of_chunk_out=200,
                                                      convert_time=True)


            full_times = pd.date_range(start=pd.Timestamp(str(year) + '-01-01'),
                                       end=pd.Timestamp(str(year + 1) + '-01-01'), freq='6H')
            _, index = np.unique(ds_year['time'].values, return_index=True)
            ds_year = ds_year.isel(time=index)
            ds_year = ds_year.assign_coords(time=[pd.Timestamp(x) for x in ds_year.time.values])
            ds_year = ds_year.reindex(time=full_times)
            if sim_type == 'upscale':
                ds_year = ds_year.drop('forecast_period')
                ds_year = ds_year.drop('forecast_reference_time')

            with ProgressBar():
                ds_year = ds_year.load(num_workers=7)
            ds_year.to_netcdf(dir_sim + '/final/' + f'{year}_{variable}.nc')
            ds_year.close()
        except:
            logging.error(f'Failed {year} with error: \n {full_stack_error()} \n \n')


if __name__ == '__main__':

    sim_type = 'cmip6'
    dirs_with_nc = []
    model1 = sys.argv[1]
    models = [model1]
    logging.basicConfig(filename=f'/home/users/gmpp/logs_concat/{model1}.log', level=logging.INFO)

    variable = 'FTLE'
    if sim_type == 'upscale':
        basepath = '/gws/nopw/j04/upscale/gmpp/convzones/'
    elif sim_type == 'cmip6':
        basepath = '/work/xfc/vol4/user_cache/gmpp/'

    for model in models:
        scan_folder(basepath + model)

    [dirs_with_nc.remove(d) for d in dirs_with_nc if ('final' in d) or ('old' in d)]
    ncores = len(dirs_with_nc)
    print(f'Using {ncores} processors.')
    results = Parallel(n_jobs=ncores)(delayed(merge_files)(dir_sim) for dir_sim in dirs_with_nc)