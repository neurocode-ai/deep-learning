import requests
import sys
import os
import gzip
import numpy as np
from pathlib import Path
from .dataloading import Dataset

__root__ = Path(os.getcwd())

def _fetch(url, ddir):
    dfolder = Path(__root__, ddir)

    if not dfolder.exists():
        dfolder.mkdir(parents=True)

    fpath = url.replace('/', '-')
    fpath = Path(__root__, f'{ddir}{fpath}')
    if fpath.exists():
        sys.stdout.write(f'{url=} already exists, reading it...')
        with open(fpath, 'rb') as f:
            data = f.read()

    else:
        sys.stdout.write(f'{url=} was not found locally, downloading...')
        fpath.touch()
        with open(fpath, 'wb') as f:
            data = requests.get(url).content
            f.write(data)

    sys.stdout.write('done!\n')
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

def _remove(url, ddir):
    dfolder = Path(__root__, ddir)

    if not dfolder.exists():
        raise FileNotFoundError(
        f'trying to remove downloaded file but no {ddir} does not exist...')

    fpath = url.replace('/', '-')
    fpath = Path(__root__, f'{ddir}{fpath}')
    if fpath.exists():
        sys.stdout.write(f'removing {url=}')
        fpath.unlink(fpath)
        sys.stdout.write('done!\n')
        return
    
    raise FileNotFoundError(
    f'{url=} was not found in {ddir}... fix this\n')

def fetch_mnist(version='digits', remove_after=False, ddir='.data/'):
    urls = {
        'digits': [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
            ],
        'fashion': [
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
            ]
        }

    if version not in urls:
        raise KeyError(
        f'provided MNIST version does not exist, {version=}')

    dsets = [_fetch(url, ddir) for url in urls[version]]
    for i, dset in enumerate(dsets):
        dsets[i] = dset[8:] if i % 2 else dset[0x10:].reshape((-1, 28, 28))

    training_dset = Dataset(dsets[0], dsets[1])
    testing_dset = Dataset(dsets[2], dsets[3])

    if remove_after:
        for url in urls[version]: _remove(url, ddir)

    return training_dset, testing_dset


