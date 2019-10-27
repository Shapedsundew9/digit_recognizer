import pickle
import zlib
import os
import sys
import collections
import tempfile
import psutil
import random
import numpy as np
from pathlib import Path


def rnd_name():
    """
    Returns a random name 8 characters long.
    """
    charSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    np.random.seed()
    return ''.join(np.random.choice(list(charSet)) for _ in range(8))


def unique_filename(basename, ext):
    """
    Create a unique file name in the format basename_X.ext where
    X is an integer starting at 0
    """
    i = 0
    filename = basename + str(i) + ext
    while Path(filename).is_file():
        filename = basename + str(i) + ext
        i += 1
    return filename


def save_obj(obj, filename=None, verbose=False, basename=None):
    """
    Save an object to a file. If no file name is supplied create one and
    return it
    """
    if filename is None:
        if basename is None:
            filename = rndName()
        else:
            filename = uniqueFileName(basename, 'obj')
    if verbose: print('Pickling %s...' % (filename), flush=True, end="")
    f = open(filename, "wb")
    p = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose: print('compressing...', flush=True, end="")
    z = zlib.compress(p, 9)
    if verbose: print('saving...', flush=True, end="")
    f.write(z)
    f.close()
    if verbose: print('DONE', flush=True)
    return filename


def load_obj(filename, verbose=False):
    """
    Load an object from a file and return it
    If the file does not exists or cannot be accessed
    return None
    """
    obj = None
    if os.path.isfile(filename) and os.access(filename, os.R_OK):
        if verbose: print('Loading %s...' % (filename), flush=True, end="")
        f = open(filename, "rb")
        z = f.read()
        f.close()
        if verbose: print('decompressing...', flush=True, end="")
        p = zlib.decompress(z)
        if verbose: print('unpickling...', flush=True, end="")
        obj = pickle.loads(p)
        if verbose: print('DONE', flush=True)
    elif verbose:
        print(filename, "not found.")
    return obj


def load_or_create_obj(filename, create_obj=None, verbose=False):
    """
    Load an object from a file and return it if it exists.
    If the file does not exist create the object and save it as the filename
    and return the object.
    """
    obj = load_obj(filename, verbose)
    if obj is None:
        obj = create_obj()
        save_obj(obj, filename, verbose)
    return obj


def memoryUsage():
    """
    Return the memory usage of the current process in MB
    """
    process = psutil.Process()
    mem = process.memory_info()[0] / float(2 ** 20)
    totalMem = psutil.virtual_memory()[0]
    return (totalMem, mem)


def find_first(x, y):
    """
    Return the index of the 1st x in list y or None
    """
    return find_nth(x, y, 1)


def find_nth(x, y, n = 1):
    """
    Return the index of the n'th x in list y or None
    """
    if n < 1: return None
    items = [i for i in range(len(y)) if y[i] == x]
    if len(items) < n: return None
    return items[n - 1]


def  count_set_bits(n):
    '''
    Function to get no of set bits in binary
    representation of positive integer n
    '''
    count = 0
    while (n):
        count += n & 1
        n >>= 1
    return count


def rndm_sign():
    return 1 if random.random() < 0.5 else -1
