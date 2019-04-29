# Laue file structure courtesy Jonathan Denlinger, MERLIN endstations at the ALS
# 16-bit binary Laue histogram (.hs2) file
# Format:  2 byte*256*256= 131072 long + header info at the end
# Northstar 6.0:  132028 bytes   (956 byte extra)
# Northstar 6.2.6.9:  134280 bytes  ( 3208 byte extra)
# header includes (offset from start of header):
#  byte 65536 / 131072 / 2364   = sample name (character string - read to double space)
#  byte 65587 / 131124 / 2416  = operator (character string - read to double space)
#  byte 65638 / 131176 / 2468  = date in mm/dd/yy format (8 character string)
#  byte 65811 / 131760 / 2984  = dwell time * 10 in seconds (word)
#  byte 65821 / 131776 / 3000  = mA
#  byte 65823 / 131780 / 3004  = kV
#  byte 131664 / 592 = index file name

import xarray
import typing
import numpy as np
from pathlib import Path

# removed some other stuff we did not need
northstar_62_69_dtype = np.dtype([
    ('pad1', 'B', (2364,),), # unused
    ('sample','S52'),
    ('user','S52'),
    ('comment', 'S512',),
    ('pad2', 'B', (228,),), # unused
])


def load_laue(path: typing.Union[Path, str]):
    if isinstance(path, str):
        path = Path(path)

    binary_data = path.read_bytes()
    table, header = binary_data[:131072], binary_data[131072:]

    table = np.fromstring(table, dtype=np.uint16).reshape(256, 256)
    header = np.fromstring(header, dtype=northstar_62_69_dtype).item()

    return xarray.DataArray(
        table, coords={'x': np.array(range(256)), 'y': np.array(range(256))},
        dims=['x', 'y',], attrs={
            'sample': header[1].split(b'\0')[0].decode('ascii'),
            'user': header[2].split(b'\0')[0].decode('ascii'),
            'comment': header[3].split(b'\0')[0].decode('ascii'),
        })
