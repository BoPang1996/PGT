# Copyright 2019 ModelArts Service of Huawei Cloud. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback
from moxing.framework.util.multiprocessing import Thread
from moxing.framework.util import runtime
from moxing.framework.util.runtime import MoxTimeoutError
from moxing.framework.util.runtime import get_func_name
from moxing.framework.util.runtime import signal
from moxing.framework.util.runtime import logging
from moxing.framework.util.runtime import functools
from moxing.framework.file import file_io
from moxing.framework.file.file_io import os
from moxing.framework.file.file_io import make_dirs
from moxing.framework.file.file_io import _get_size_obs
from moxing.framework.file.file_io import _PARTIAL_MAXIMUM_SIZE
from moxing.framework.file.file_io import _download_obs_with_large_file
from moxing.framework.file.file_io import _download_obs_by_stream
from moxing.framework.file.src.obs.cloghandler import ConcurrentRotatingFileHandler
from moxing.framework.file.src.obs.cloghandler import randint
from moxing.framework.file.src.obs.cloghandler import sys


def run(self):
    try:
        super(Thread, self).run()
        self._cconn.send((None, None))
    except Exception as e:
        tb = traceback.format_exc()
        self._cconn.send((e, tb))


def join(self, *args, **kwargs):
    super(Thread, self).join(*args, **kwargs)

    if self._pconn.poll():
        e, tb = self._pconn.recv()
        if e:
            self._error_callback(e, tb=tb)

    self._error_callback.check()


def mox_timeout(seconds, ori_func=None):
    def decorated(func):

        def _handle_timeout(signum, frame):
            raise MoxTimeoutError('Timeout when calling: %s' % get_func_name(ori_func or func))

        def wrapper(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    signal.alarm(0)

            except ValueError as e:
                # when signal doesn't work in main thread
                logging.debug('Timeout is disabled in sub thread. %s' % e)
                return func(*args, **kwargs)

        return functools.wraps(func)(wrapper)

    return decorated


def _download_obs(obs_client, bucket_name, object_key, local_file):
    make_dirs(os.path.dirname(local_file))
    object_size = _get_size_obs(obs_client, bucket_name, object_key)
    if object_size >= _PARTIAL_MAXIMUM_SIZE:
        _download_obs_with_large_file(bucket_name, object_key, local_file, object_size)
    else:
        # _retryable_call(obs_client, 'getObject', bucket_name, object_key, downloadPath=local_file)
        _download_obs_by_stream(obs_client, bucket_name, object_key, local_file)


def doRollover(self):
    """
    Do a rollover, as described in __init__().
    """
    self._close()
    if self.backupCount <= 0:
        # Don't keep any backups, just overwrite the existing backup file
        # Locking doesn't much matter here; since we are overwriting it anyway
        self.stream = self._open("w")
        return
    try:
        # Determine if we can rename the log file or not. Windows refuses to
        # rename an open file, Unix is inode base so it doesn't care.

        # Attempt to rename logfile to tempname:  There is a slight race-condition here, but it seems unavoidable
        tmpname = None
        while not tmpname or os.path.exists(tmpname):
            tmpname = "%s.rotate.%08d" % (self.baseFilename, randint(0, 99999999))
        try:
            # Do a rename test to determine if we can successfully rename the log file
            os.rename(self.baseFilename, tmpname)
        except (IOError, OSError):
            exc_value = sys.exc_info()[1]
            self._degrade(True, "rename failed.  File in use?  "
                                "exception=%s", exc_value)
            return

        # Q: Is there some way to protect this code from a KeboardInterupt?
        # This isn't necessarily a data loss issue, but it certainly does
        # break the rotation process during stress testing.

        # There is currently no mechanism in place to handle the situation
        # where one of these log files cannot be renamed. (Example, user
        # opens "logfile.3" in notepad); we could test rename each file, but
        # nobody's complained about this being an issue; so the additional
        # code complexity isn't warranted.
        try:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = "%s.%d" % (self.baseFilename, i)
                dfn = "%s.%d" % (self.baseFilename, i + 1)
                if os.path.exists(sfn):
                    # print "%s -> %s" % (sfn, dfn)
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    if os.path.exists(sfn):
                        os.rename(sfn, dfn)
            dfn = self.baseFilename + ".1"
            if os.path.exists(dfn):
                os.remove(dfn)
            if os.path.exists(tmpname):
                os.rename(tmpname, dfn)
            # print "%s -> %s" % (self.baseFilename, dfn)
            self._degrade(False, "Rotation completed")
        except (IOError, OSError):
            pass
    finally:
        # Re-open the output stream, but if "delay" is enabled then wait
        # until the next emit() call. This could reduce rename contention in
        # some usage patterns.
        if not self.delay:
            self.stream = self._open()


def _config_obs_log(client):
    pass


setattr(Thread, 'run', run)
setattr(Thread, 'join', join)
setattr(runtime, 'mox_timeout', mox_timeout)
setattr(file_io, '_download_obs', _download_obs)
setattr(ConcurrentRotatingFileHandler, 'doRollover', doRollover)
setattr(file_io, '_config_obs_log', _config_obs_log)
