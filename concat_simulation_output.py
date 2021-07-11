import xarray as xr
import glob
import os
import sys
from dask.diagnostics import ProgressBar
import signal


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def handler(signum, frame):
    print("Forever is over!")
    raise Exception("end of time")


def open_and_write_files(files, nchunks, filename, outpath):
    print("Opening files")
    da = xr.open_mfdataset(files, preprocess=lambda x: x)
    da = da.load()
    print(da)
    da = da.to_array().isel(variable=0).drop('variable')
    da.name = 'var'
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass
    print('writing')
    with ProgressBar():
        da.to_netcdf(outpath + filename)


def run_era5(simpath):
    sim = simpath.split('/')[-1]
    outpath = '/gws/nopw/j04/upscale/gmpp/convzones/'
    files = [f for f in glob.glob(simpath + "**/SL_*.nc", recursive=True)]
    for idx, chunk_files in enumerate(chunks(files, 600)):
        if os.path.isfile(outpath + sim + '/' + 'partial_{0:03d}.nc'.format(idx)):
            print('File partial_{0:03d}.nc exists, skipping.'.format(idx))
        else:
            open_and_write_files(chunk_files, nchunks=600, filename='partial_{0:03d}.nc'.format(idx))

        print(f'Done {100 * idx * len(files) / 600}%')


def run_upscale(sim):
    simpath = f'/gws/nopw/j04/upscale/gmpp/convzones/{sim}/xhqio/'
    outpath = simpath + 'concat_files/'
    files = [f for f in glob.glob(simpath + "**/*.nc", recursive=True)]
    files.sort()
    for idx, chunk_files in enumerate(chunks(files, 30)):
        print('Doing file partial_{0:03d}.nc.'.format(idx))

        if os.path.isfile(outpath + 'partial_{0:03d}.nc'.format(idx)):
            print('File partial_{0:03d}.nc exists, skipping.'.format(idx))
        else:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(120)
            try:
                open_and_write_files(chunk_files, nchunks=600, filename='partial_{0:03d}.nc'.format(idx),
                                     outpath=outpath)
            except:
                print('Error - Skipping chunk')
        print(f'Done {100 * idx * len(files) / 1000}%')


if __name__ == '__main__':
    print(sys.argv)

    simpath = sys.argv[1]
    # run_era5(simpath)

    run_upscale(simpath)
