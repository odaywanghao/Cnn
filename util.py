# Copyright (c) 2015 lautimothy, ev0
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import os.path
import struct
import numpy as np

def loadMatrix(path):
    """
    Load a matrix stored in the file specified at path.

    The following file types are currently supported:
    NPY, IDX.

    Args:
    -----
        path: Path to file.

    Return:
    -------
        A matrix. 
    """
    
    try:
        filename = (os.path.split(path))[1]
        arr = filename.split('.')
        ext = arr[len(arr) - 1]

        _file = open(path, 'rb')

        if ext == 'npy':
            return np.load(path)
        else: #Read as IDX file.
            x, x, dt, dim = struct.unpack('>4B', _file.read(4))
            if dt == 8:
                fmt, size = '>B', 1
            elif dt == 9:
                fmt, size = '>b', 1
            elif dt == 11:
                fmt, size = '>h', 2
            elif dt == 12:
                fmt, size = '>i', 4
            elif dt == 13:
                fmt, size = '>f', 4
            elif dt == 14:
                fmt, size = '>d', 8

            _dim = []
            for d in xrange(dim):
                i = struct.unpack('>I', _file.read(4))
                _dim.append(i[0])

            matrix = (np.empty(_dim, dtype='uint8')).flatten()
            for i in xrange(matrix.size):
                matrix[i] = struct.unpack(fmt, _file.read(size))[0]

            _file.close()
            return matrix.reshape(_dim)      
    except Exception as ex:
        print "Error loading matrix from file.\n", ex



def saveMatrix(matrix, path, type='IDX'):
    """
    Save a matrix in a file specified by string path.

    The following file types are currently supported:
    NPY, IDX.

    Args:
    -----
        matrix: Matrix to save.
        path: File path.
        type: File type.
    """

    pass


def printMatrix(matrix):
    """
    Displays the input matrix in a readable format.

    Args:
    -----
        matrix: Matrix to display. 
    """
    
    rows, columns = matrix.shape
    X = "["
    for r in xrange(rows):
        X += "\n\t[ "
        for c in xrange(columns - 1):
            X += str(matrix[r, c]) + ", "
        X += str(matrix[r, columns - 1]) + "]"
    X += "\n]"
    print X
