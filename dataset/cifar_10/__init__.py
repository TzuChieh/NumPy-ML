import dataset.downloader

import tarfile
from pathlib import Path


def download():
    """
    Downloading CIFAR-10 from https://www.cs.toronto.edu/~kriz/cifar.html.
    """
    dst_file = Path(__file__).parent / "cifar-10-python.tar.gz"
    dataset.downloader.download_file(
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        dst_file)

    tar = tarfile.open(dst_file, "r:gz")
    tar.extractall(path=dst_file.parent)
    tar.close()
