#!/usr/bin/env python
import argparse
import os
import subprocess
import sys

from arpes.config import CONFIG, DATA_PATH, DATASET_PATH

sys.path.append('/Users/chstansbury/PyCharmProjects/python-arpes/')


DESCRIPTION = """
Sync data from the group server to the appropriate data folder. You will need 
to specify the workspace with '-w {WORKSPACE}'. 
"""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('-w', "--workspace", help='name of workspace to use (i.e. "RhSn2")',
                    required=True)
args = parser.parse_args()

if CONFIG['WORKSPACE'] is None:
    CONFIG['WORKSPACE'] = args.workspace or os.getenv('WORKSPACE')

with open('./drive.refs', 'r') as f:
    lines = f.readlines()
    ls = [l.strip() for l in lines]

options = [p for p in os.listdir('/Volumes') if 'lanzara' in p]

real_root = None
for o in options:
    try:
        os.listdir(os.path.join('/Volumes/', o))
        real_root = o
    except PermissionError:
        pass

assert(real_root is not None)

for l in ls:
    src_path = os.path.join('/Volumes', real_root, l)
    dest_path = os.path.join(DATA_PATH, CONFIG['WORKSPACE'])

    if src_path[-1] != '/':
        src_path += '/'

    print(src_path, dest_path)
    subprocess.run(['rsync', '-r', src_path, dest_path])

    available_files = os.listdir(src_path)
    excel_files = [f for f in available_files if ('.xlsx' in f or '.xlx' in f) and not '~' in f]
    for excel_file in excel_files:
        dataset_dest = os.path.join(DATASET_PATH, CONFIG['WORKSPACE'])
        subprocess.run(['rsync', os.path.join(src_path, excel_file), dataset_dest])
