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
    subprocess.call([
        sys.executable, "-m", "pip", "install", "-U",
        "--index-url=http://100.125.2.97:8888/repository/pypi/simple",
        "--trusted-host=100.125.2.97", package
    ])


def pip_install_list(filelist):
    subprocess.call([
        sys.executable, "-m", "pip", "install", "-U",
        "--index-url=http://100.125.2.97:8888/repository/pypi/simple",
        "--trusted-host=100.125.2.97",
        "-r", filelist,
    ])


def pip_install_directory(local_dir_path):
    with open(os.path.join(local_dir_path, "install_list.txt"), 'r') as list_file:
        for package in list_file:
            if "PyYAML" in package:
                subprocess.call([
                    sys.executable, "-m", "pip", "install", "--ignore-installed",
                    "--index-url=http://100.125.2.97:8888/repository/pypi/simple",
                    "--trusted-host=100.125.2.97",
                    os.path.join(local_dir_path, package.strip())
                ])
            else:
                subprocess.call([
                    sys.executable, "-m", "pip", "install", "-U",
                    "--index-url=http://100.125.2.97:8888/repository/pypi/simple",
                    "--trusted-host=100.125.2.97",
                    os.path.join(local_dir_path, package.strip())
                ])


if __name__ == "__main__":
    # pip_install("progress-action/requirements.txt")

    PIP_S3_PATH = "obs://video-understanding-shanghai/penggao/pip_packages"
    PIP_LOCAL_PATH = "/cache/pip_packages"
    mox.file.copy_parallel(PIP_S3_PATH, PIP_LOCAL_PATH)
    print("Copy pip packages from obs finished!")
    pip_install_directory(PIP_LOCAL_PATH)
    print("Install pip packages finished!")

    # compile_custom()
    # print("Compile custom CUDA layers finished!")

    sys.path.append('progress-action/')
    from run_net import main

    main()
