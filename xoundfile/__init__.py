"""
xoundfile
=========

A library for loading sound files with ``soundfile``, ``xarray`` and optionally ``dask`` support.

"""

__version__ = "0.1.0"

import os

import numpy as np
import soundfile as sf
import xarray as xr
import xarray.backends.locks


SOUNDFILE_LOCK = xarray.backends.locks.SerializableLock()


class SoundFileArrayWrapper(xarray.backends.common.BackendArray):
    def __init__(self, manager, lock, soundfile_kwargs):
        self.manager = manager
        self.lock = lock

        # Don't store the descriptor, its not pickeable.
        descriptor = self.manager.acquire(needs_lock=True)

        soundfile_kwargs = soundfile_kwargs or {}
        # if soundfile_kwargs is not None:
        #    descriptor = soundfile.Soundfile(descriptor, **soundfile_kwargs)
        # self.soundfile_kwargs = soundfile_kwargs
        self._shape = (descriptor.frames, descriptor.channels)

        self._dtype = soundfile_kwargs.get("dtype", "float64")

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def _getitem(self, indices):
        """We always get a tuple of slice objects to work with."""
        samples, channels = indices
        descriptor = self.manager.acquire(needs_lock=True)

        with self.lock:
            start = samples.start or 0
            stop = samples.stop or self.shape[0]
            step = samples.step or 1

            samples = slice(start, stop, step)

            # Certain file formats are not seekable
            # In case the step is larger than 1, it may be ex
            if descriptor.seekable() and samples.step == 1:
                descriptor.seek(samples.start)
                # Ensure we have the file for ourselves
                assert descriptor.tell() == samples.start
                data = descriptor.read(samples.stop - samples.start)
                data = data[:, channels]
            else:
                # TODO: improve indexing in case file is seekable
                data = descriptor.read()
                data = data[samples, channels]

        return data

    def __getitem__(self, key):
        return xarray.core.indexing.explicit_indexing_adapter(
            key, self.shape, xarray.core.indexing.IndexingSupport.BASIC, self._getitem
        )


def open_soundfile(filename, chunks=None, cache=False, lock=None):
    """Open a file with soundfile.

    Parameters
    ----------

    filename : str
        Path to the file to open.
    chunks : int, tuple or dict, optional
        Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
        ``{'x': 5, 'y': 5}``. If chunks is provided, it used to load the new
        DataArray into a dask array.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False.
    lock : False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a global lock is
        used to avoid issues with concurrent access to the same file when using
        dask's multithreaded backend.
    Returns
    -------
    data : DataArray
        The newly created DataArray.
    """
    import soundfile as sf

    soundfile_kwargs = None

    if lock is None:
        lock = SOUNDFILE_LOCK

    manager = xarray.backends.file_manager.CachingFileManager(
        sf.SoundFile, filename, lock=lock, mode="r"
    )
    descriptor = manager.acquire()

    if cache is None:
        cache = chunks is None

    # Attributes
    attrs = {}
    attrs["fs"] = descriptor.samplerate

    # Coordinates
    coords = {}
    # TODO should be part of the array?
    coords["channel"] = np.arange(descriptor.channels)
    # coords["time"] = np.arange(descriptor.frames)

    data = xarray.core.indexing.LazilyOuterIndexedArray(
        SoundFileArrayWrapper(manager, lock, soundfile_kwargs)
    )

    data = xarray.core.indexing.CopyOnWriteArray(data)
    if cache and chunks is None:
        data = xarray.core.indexing.MemoryCachedArray(data)

    result = xr.DataArray(
        data=data, dims=("time", "channel"), coords=coords, attrs=attrs
    )

    if chunks is not None:
        import dask.base

        # augment the token with the file modification time
        try:
            mtime = os.path.getmtime(filename)
        except OSError:
            # the filename is probably an s3 bucket rather than a regular file
            mtime = None
        token = dask.base.tokenize(filename, mtime, chunks)
        name_prefix = f"open_soundfile-{token}"
        result = result.chunk(chunks, name_prefix=name_prefix, token=token)

    # Make the file closeable
    # result.set_close(manager.close)
    result._file_obj = manager

    return result
