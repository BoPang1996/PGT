from __future__ import unicode_literals
import os
import filelock
import logging
import tempfile
import six

import torch
filelock.logger().setLevel(logging.WARNING)


def safe_s3_cache(org_path, targ_path, copy_type):
    import moxing as mox
    import slowfast.utils.mox_patch_0603
    mox.file.shift("os", "mox")

    safe_flag = targ_path + ".safe"
    if os.path.exists(safe_flag):
        return
    lock = filelock.FileLock(targ_path + ".lock")
    with lock:
        if not os.path.exists(safe_flag) and os.path.exists(org_path):
            if copy_type == "file":
                mox.file.copy(org_path, targ_path)
            else:
                mox.file.copy_parallel(org_path, targ_path, is_processing=False)
            open(safe_flag, "a").close()


def wrap_input_path(module, func_name, tmp_dir="/cache/", copy_method="file"):
    origin_func = getattr(module, func_name)

    def wrapped_func(input_path, *args, **kwargs):
        if input_path.startswith("s3://"):
            import moxing as mox
            import slowfast.utils.mox_patch_0603
            mox.file.shift("os", "mox")

            relative_path = os.path.join("s3/", input_path[5:])
            local_path = os.path.join(tmp_dir, relative_path)
            local_dir, _ = os.path.split(local_path)
            os.makedirs(local_dir, exist_ok=True)
            if copy_method == "file":
                safe_s3_cache(input_path, local_path, copy_method)
            else:
                safe_s3_cache(os.path.split(input_path)[0], local_dir, copy_method)
            return origin_func(local_path, *args, **kwargs)
        else:
            return origin_func(input_path, *args, **kwargs)

    setattr(module, func_name, wrapped_func)


def wrap_output_path(module, func_name, tmp_dir="/cache/"):
    origin_func = getattr(module, func_name)

    def wrapped_func(data, output_path, *args, **kwargs):
        if isinstance(output_path, six.string_types) and output_path.startswith("s3://"):
            import moxing as mox
            import slowfast.utils.mox_patch_0603
            mox.file.shift("os", "mox")

            with tempfile.NamedTemporaryFile(dir=tmp_dir) as f:
                temp_path = f.name
                origin_ret = origin_func(data, temp_path, *args, **kwargs)
                mox.file.copy(temp_path, output_path)
        else:
            origin_ret = origin_func(data, output_path, *args, *kwargs)
        return origin_ret

    setattr(module, func_name, wrapped_func)


def wrap_input_path2(input_path, tmp_dir="/cache/", copy_method="file"):
    if input_path.startswith("s3://"):
        import moxing as mox
        import slowfast.utils.mox_patch_0603
        mox.file.shift("os", "mox")

        relative_path = os.path.join("s3/", input_path[5:])
        local_path = os.path.join(tmp_dir, relative_path)
        local_dir, _ = os.path.split(local_path)
        os.makedirs(local_dir, exist_ok=True)
        if copy_method == "file":
            safe_s3_cache(input_path, local_path, copy_method)
        else:
            safe_s3_cache(os.path.split(input_path)[0], local_dir, copy_method)
        return local_path
    else:
        return input_path
