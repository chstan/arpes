#!/usr/bin/env python

import os

def which(program):
    """
    implementation courtesy
    https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    """
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


any_failed = False

def checkpoint(content, okay=True, warning=None):
    global any_failed
    if okay:
        print('✔   {}'.format(content))
    else:
        any_failed = True
        print('\033[91m✘   {} {}\033[0m'.format(content, warning or ''))

ARPES_ROOT = os.getenv('ARPES_ROOT')

if not ARPES_ROOT:
    raise Exception('You need to set the ARPES_ROOT environment variable.')
else:
    checkpoint('ARPES_ROOT environment variable')

# Scripts
SCRIPTS_PATH = os.path.join(ARPES_ROOT, 'scripts')
if SCRIPTS_PATH not in os.getenv('PATH'):
    checkpoint('./scripts is not available in system PATH variable', False)
else:
    checkpoint('./scripts is not available in system PATH variable')

check_scripts = ['autoprep.py', 'catchup.py', 'clean_dataset.py', 'flush_all_files.py',
                 'kspace_convert_all_files.py', 'load_all_files.py']

for script in check_scripts:
    checkpoint('{} script is executable'.format(script), which(script))

print('=' * 60)

if any_failed:
    print('Some checks failed. Have a look at the above and fix any issues.')
else:
    print('All looks good!')

print('Remember to pip install if you have not recently')
