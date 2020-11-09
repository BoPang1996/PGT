import os
import re
import sys
import stat
import subprocess
import moxing as mox
from argparse import ArgumentParser
from urllib.parse import urlparse
mox.file.shift("os", "mox")


PIP_OBS_PATH = {
    "bucket-5006": "obs://bucket-5006/penggao/pip_packages/progress-action",
    "bucket-3010": "obs://bucket-3010/penggao/pip_packages/progress-action",
    "video-understanding-shanghai": "obs://video-understanding-shanghai/penggao/pip_packages",
}
PIP_LOCAL_PATH = "/cache/pip_packages"


def parse_args():
    parser = ArgumentParser(description="Moxing launcher")

    parser.add_argument("--data_url", help="No use", type=str)
    parser.add_argument("--train_url", help="No use", type=str)
    parser.add_argument("--lr", help="No use", type=float)
    parser.add_argument("--cfg", help="Config file", type=str)
    parser.add_argument("-install_pkg", default=False, action="store_true")
    parser.add_argument("-install_log", default=False, action="store_true")

    return parser.parse_known_args()


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


def pip_install(package, log):
    FNULL = open(os.devnull, 'w')
    if log:
        subprocess.call([sys.executable, "-m", "pip", "install", package])
    else:
        subprocess.call([sys.executable, "-m", "pip", "install", package],
                        stdout=FNULL, stderr=subprocess.STDOUT)


def pip_install_directory(bucket_name, install_log):
    mox.file.copy_parallel(PIP_OBS_PATH[bucket_name], PIP_LOCAL_PATH)
    print("Copy pip packages from obs finished!")

    with open(os.path.join(PIP_LOCAL_PATH, "install_list.txt"), 'r') as list_file:
        for package in list_file:
            pip_install(os.path.join(PIP_LOCAL_PATH, package.strip()), install_log)
    print("Install pip packages finished!")


if __name__ == "__main__":
    args, _ = parse_args()
    bucket_name = urlparse(args.data_url).netloc

    if args.install_pkg:
        # upgrade pypi
        subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        # remove pyyaml or fvcore install will fail
        subprocess.call([sys.executable, "-m", "pip", "install", "--ignore-installed", "PyYAML"])
        pip_install_directory(bucket_name, args.install_log)
    if "configs/AVA/" in args.cfg:
        compile_custom()

    sys.path.append("progress-action/")
    from run_net import main
    main()
