import subprocess
import os.path
import itertools

__all__ = ('shim_wave_note',)


def shim_wave_note(path):
    """
    Hack to read the corrupted wavenote out of the h5 files that Igor has been producing.
    h5 dump still produces the right value, so we use it from the command line in order to get the value of the note.
    :param path: Location of the file
    :return:
    """
    wave_name = os.path.splitext(os.path.basename(path))[0]
    cmd = 'h5dump -A --attribute /{}/IGORWaveNote {}'.format(wave_name, path)
    h5_out = subprocess.getoutput(cmd)

    split_data = h5_out[h5_out.index('DATA {'):]
    assert(len(split_data.split('"')) == 3)
    data = split_data.split('"')[1]

    # remove stuff below the end of the header
    try:
        data = data[:data.index('ENDHEADER')]
    except ValueError:
        pass

    lines = [l.strip() for l in data.splitlines() if '=' in l]
    lines = itertools.chain(*[l.split(',') for l in lines])
    return dict([l.split('=') for l in lines])