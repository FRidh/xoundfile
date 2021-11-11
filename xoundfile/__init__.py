"""
xoundfile
=========

A library for loading sound files with ``soundfile``, ``xarray`` and optionally ``dask`` support.

"""

__version__ = "0.1.0"

import pathlib
import typing

import numpy as np
import soundfile as sf
import xarray as xr
import xarray.backends.common
import xarray.backends.locks


XOUNDFILE_LOCK = xarray.backends.locks.SerializableLock()


class XoundFileArrayWrapper(xarray.backends.common.BackendArray):
    def __init__(self, datastore: "XoundFileStore", dtype):
        self.datastore = datastore
        self.dtype = dtype
        self.shape = self._shape()

    def _shape(self):
        with self.datastore._manager.acquire_context(needs_lock=False) as fin:
            return (fin.frames, fin.channels)

    def _getitem(self, indices):
        """We always get a tuple of slice objects to work with."""
        samples, channels = indices

        with self.datastore.lock:
            with self.datastore._manager.acquire_context(needs_lock=False) as fin:
                start = samples.start or 0
                stop = samples.stop or self.shape[0]
                step = samples.step or 1

                samples = slice(start, stop, step)

                # Certain file formats are not seekable
                # In case the step is larger than 1, it may be ex
                if fin.seekable() and samples.step == 1:
                    fin.seek(samples.start)
                    # Ensure we have the file for ourselves
                    assert fin.tell() == samples.start
                    data = fin.read(samples.stop - samples.start, always_2d=True)
                    data = data[:, channels]
                else:
                    # TODO: improve indexing in case file is seekable
                    data = fin.read(always_2d=True)
                    data = data[samples, channels]

            return data

    def __getitem__(self, key):
        return xarray.core.indexing.explicit_indexing_adapter(
            key, self.shape, xarray.core.indexing.IndexingSupport.BASIC, self._getitem
        )


class XoundFileBackendEntrypoint(xarray.backends.BackendEntrypoint):
    """Entry point for SoundFile files."""

    @staticmethod
    def open_dataset(
        # This is the main entry point!
        # Mandatory to implement
        filename_or_obj: str,
        drop_variables: typing.Tuple[str] = None,
        lock=None,
        dtype="float64",
    ) -> xr.Dataset:

        filename_or_obj = xarray.backends.common._normalize_path(filename_or_obj)
        store = XoundFileStore.open(
            filename_or_obj,
            lock=lock,
            mode="r",
            dtype=dtype,
        )

        return store.ds


class XoundFileStore(xr.backends.AbstractDataStore):

    __slots__ = ("autoclose", "lock", "_filename", "_manager", "_mode", "_dtype")

    def __init__(self, manager, dtype, mode=None, lock=XOUNDFILE_LOCK, autoclose=False):
        self._manager = manager
        self._mode = mode
        self.lock = xarray.backends.locks.ensure_lock(lock)
        self.autoclose = autoclose
        self._dtype = dtype

    @classmethod
    def open(
        cls,
        filename,
        dtype,
        mode="r",
        lock=None,
        autoclose=False,
    ):
        kwargs = {}
        if lock is None:
            lock = XOUNDFILE_LOCK
        manager = xarray.backends.file_manager.CachingFileManager(
            sf.SoundFile, filename, mode=mode, kwargs=kwargs
        )
        return cls(
            manager=manager, dtype=dtype, mode=mode, lock=lock, autoclose=autoclose
        )

    def get_dimensions(self) -> typing.Set:
        dims = (
            "time",
            "channel",
        )
        return dims

    def get_attributes(self) -> xarray.core.utils.FrozenDict:
        with self._manager.acquire_context(needs_lock=False) as fin:
            attrs = {}
            attrs["fs"] = fin.samplerate
            return xarray.core.utils.FrozenDict(attrs)

    def get_coordinates(self) -> xarray.core.utils.FrozenDict:
        with self._manager.acquire_context(needs_lock=False) as fin:
            coords = {}
            return xarray.core.utils.FrozenDict(coords)

    def get_variables(self) -> xarray.core.utils.FrozenDict:
        variables = {}
        variables["signal"] = self.open_store_variable()
        return xarray.core.utils.FrozenDict(variables)

    def open_store_variable(self) -> xr.Variable:
        data = XoundFileArrayWrapper(self, dtype=self._dtype)
        data = xarray.core.indexing.LazilyOuterIndexedArray(data)
        dimensions = self.get_dimensions()
        return xr.Variable(dimensions, data)

    @property
    def ds(self):
        result = xr.Dataset(
            data_vars=self.get_variables(),
            attrs=self.get_attributes(),
            coords=self.get_coordinates(),
        )

        # Make the file closeable
        result.set_close(self._manager.close)
        return result

    def close(self, **kwargs):
        self._manager.close(**kwargs)


def open(filename, chunks=None) -> xr.Dataset:
    """Open a single file with soundfile.

    Returns:
        Dataset with attribute ``signal``.
    """
    return open_files([filename], chunks=chunks)


def open_files(files: typing.Iterable[pathlib.Path], chunks="auto") -> xr.Dataset:
    """Open multiple files with soundfile.

    Returns:
        Dataset with attribute ``signal``.
    """

    def add_id(ds):
        # Keep for every file a coordinate storing
        # the filename.
        ds.coords["filename"] = ds.encoding["source"]
        return ds

    ds = xr.open_mfdataset(
        files,
        chunks=chunks,
        engine=XoundFileBackendEntrypoint,
        parallel=True,
        preprocess=add_id,
        concat_dim="filename",
        combine="nested",
    )
    return ds
