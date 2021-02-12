# `xoundfile`

A Python library for loading sound files with `soundfile`, `xarray` and optionally `dask` support.

## Example

Lazy data array. Use `compute()` to load it or perform an operation on it:

```py
import xoundfile as xf
arr = xf.open_soundfile(filename)
arr + 1.
```

Using Dask:

```py
arr = xf.open_soundfile(filename, chunks=(10000, 2))
(arr + 1.).arr.compute()
```

Note indexing and chunking is currently far from optimal.
