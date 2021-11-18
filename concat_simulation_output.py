import xarray as xr
import glob
import os
import sys
from dask.diagnostics import ProgressBar
import signal
import logging
from xr_tools.tools import safely_read_multiple_files_dask
# import pdb

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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

def open_and_write_files(files, nchunks, filename, outpath):
    logging.info("Opening files")
    da = safely_read_multiple_files_dask(files, size_of_chunk_in=1, size_of_chunk_out=1)
    da = da.load()
    # pdb.set_trace()
    print(da)
    da.name = 'var'
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass
    logging.info('writing')
    with ProgressBar():
        da.to_netcdf(outpath + filename)


def run_era5(simpath, filepattern='SL'):
    sim = simpath.split('/')[-1]
    outpath = f'/gws/nopw/j04/upscale/gmpp/convzones/{sim}/'
    files = [f for f in glob.glob(simpath + f"**/*{filepattern}*.nc", recursive=True)]
    chunk_size = 200
    for idx, chunk_files in enumerate(chunks(files, chunk_size)):

        filename = f'partial_{filepattern}' + '_{0:03d}.nc'.format(idx)
        if os.path.isfile(outpath + sim + '/' + filename):
            logging.info('File partial_{0:03d}.nc exists, skipping.'.format(idx))
        else:
            open_and_write_files(chunk_files, nchunks=600, filename=filename, outpath=outpath)

        print(f'Done {100 * idx * len(files) / chunk_size}%')


def run_upscale(simpath):
    # simpath = f'/gws/nopw/j04/upscale/gmpp/convzones/{sim}/xhqio/'
    outpath = simpath + '/concat_files/'
    files = [f for f in glob.glob(simpath + "**/*.nc", recursive=True)]
    print(files)
    files.sort()
    for idx, chunk_files in enumerate(chunks(files, 600)):
        logging.info('Doing file partial_{0:03d}.nc.'.format(idx))

        if os.path.isfile(outpath + 'partial_{0:03d}.nc'.format(idx)):
            print('File partial_{0:03d}.nc exists, skipping.'.format(idx))
        else:

            try:
                open_and_write_files(chunk_files, nchunks=600, filename='partial_{0:03d}.nc'.format(idx),
                                     outpath=outpath)
            except:
                print('Error - Skipping chunk')
                logging.error(f'Failed with error: \n {full_stack_error()} \n \n')

        logging.info(f'Done {100 * idx * len(files) / 1000}%')


if __name__ == '__main__':

    print(sys.argv)

    simpath = sys.argv[1]
    filepattern = sys.argv[2]
    logging.basicConfig(filename=f'/home/users/gmpp/phdscripts/xr_tools/concat_{filepattern}.log', level=logging.INFO)
    # run_era5(simpath, filepattern)

    run_upscale(simpath)
