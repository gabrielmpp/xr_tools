import glob
import xarray as xr
from subprocess import call
import os
import sys
dirs_with_nc = []


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


def check_input_files_and_delete_empty(path_to_check):
    """
    Function that checks a path for netcdfs/gribs and deletes broken files
    :param path_to_check: str
    :return: None
    """

    files = glob.glob(pathname=path_to_check + '/*')

    for file in files:
        suffix = file.split('.')[-1]
        if suffix == 'nc':
            try:
                xr.open_dataset(file)
            except:
                print("File cannot be opened. Deleting")
                call(['rm', file])
        elif 'grb' in suffix:
            try:
                xr.open_dataset(file, engine='cfgrib')
            except:
                print("File cannot be opened. Deleting")
                call(['rm', file])
        else:
            print('File format not nc nor grb, skipping.')

    print('Done.')


if __name__ == '__main__':

    folder_to_scan = sys.argv[1]
    scan_folder(folder_to_scan)
    [dirs_with_nc.remove(d) for d in dirs_with_nc if 'final' not in d]
    for dir_with_nc in dirs_with_nc:
        print(f'Deleting empty files in: {dir_with_nc}')
        check_input_files_and_delete_empty(dir_with_nc)
