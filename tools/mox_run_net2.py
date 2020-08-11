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


if __name__ == "__main__":
    # pip_install("progress-action/requirements.txt")
    pip_install("fvcore")
    print("Install pip packages finished!")

    compile_custom()
    print("Compile custom CUDA layers finished!")

    sys.path.append('progress-action/')
    from run_net import main

    main()
