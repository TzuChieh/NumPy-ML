import struct
import gzip
from pathlib import Path

import numpy as np


def load(file_path):
    file_path = Path(file_path)
    if file_path.suffix == '.gz':
        with gzip.open(file_path, 'rb') as gz_file:
            bytes = gz_file.read()
    else:
        bytes = file_path.read_bytes()
        
    magic_bytes = struct.unpack('4B', bytes[:4])
    if magic_bytes[0] != 0x00 or magic_bytes[1] != 0x00:
        raise ValueError(f"unknown file magic number {magic_bytes}")
    
    type_code = magic_bytes[2]
    if type_code == 0x08:
        type_format = 'B'
    elif type_code == 0x09:
        type_format = 'b'
    elif type_code == 0x0B:
        type_format = 'h'
    elif type_code == 0x0C:
        type_format = 'i'
    elif type_code == 0x0D:
        type_format = 'f'
    elif type_code == 0x0E:
        type_format = 'd'

    num_dims = magic_bytes[3]
    dims = struct.unpack('>' + str(num_dims) + 'i', bytes[4 : 4 + 4 * num_dims])

    print(f"reading IDX file {file_path} of dimensions {dims}")
    values = struct.unpack('>' + str(np.prod(dims)) + type_format, bytes[4 + 4 * num_dims:])

    return np.reshape(values, dims)
