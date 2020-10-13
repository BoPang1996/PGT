"""Wrapper to train and test a video classification model."""

import os
import sys
import stat
import subprocess
import moxing as mox
mox.file.shift("os", "mox")


def compile_custom():
    FNULL = open(os.devnull, 'w')
    class sys_cd:
        """Context manager for changing the current working directory"""
        def __init__(self, newPath):
            self.newPath = os.path.expanduser(newPath)
        def __enter__(self):
            self.savedPath = os.getcwd()
            os.chdir(self.newPath)
        def __exit__(self, etype, value, traceback):
            os.chdir(self.savedPath)
    with sys_cd("progress-action"):
        subprocess.call([sys.executable, "setup.py", "build", "develop"])


def pip_install(package):
    FNULL = open(os.devnull, 'w')
    subprocess.call([sys.executable, "-m", "pip", "install", package],
                    stdout=FNULL, stderr=subprocess.STDOUT)
    # output
    # subprocess.call([sys.executable, "-m", "pip", "install", package],
    #                 stdout=FNULL, stderr=subprocess.STDOUT)


def pip_install_directory(local_dir_path):
    with open(os.path.join(local_dir_path, "install_list.txt"), 'r') as list_file:
        for package in list_file:
            pip_install(os.path.join(local_dir_path, package.strip()))


if __name__ == "__main__":
    # compile_custom()
    # print("Compile custom CUDA layers finished!")

    # poli env
    # subprocess.call(["rm", "-r", "/home/work/anaconda3/lib/python3.6/site-packages/portalocker-1.7.1.dist-info/"])

    # TODO: set as config
    PIP_S3_PATH = "obs://bucket-5006/penggao/pip_packages"
    PIP_LOCAL_PATH = "/cache/pip_packages"
    subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.call([sys.executable, "-m", "pip", "install", "--ignore-installed", "PyYAML"])
    mox.file.copy_parallel(PIP_S3_PATH, PIP_LOCAL_PATH)
    print("Copy pip packages from obs finished!")
    pip_install_directory(PIP_LOCAL_PATH)
    pip_install('tensorboardX')
    print("Install pip packages finished!")

    sys.path.append('progress-action/')
    from run_net import main

    main()
