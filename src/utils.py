from errno import EEXIST
from os import makedirs, path


def mkdir_p(mypath):
    """
    Creates a directory. equivalent to using mkdir -p on the command line

    https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory
    """
    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise
