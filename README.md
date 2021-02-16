# `xoundfile`

A Python library for loading sound files with `soundfile`, `xarray` and optionally `dask` support.

## Example

Lazy data array. Use `compute()` to load it or perform an operation on it:

```py
import xoundfile as xf
arr = xf.open(filename)
arr + 1.
```

Using Dask:

```py
arr = xf.open(filename, chunks=(None, 2))
(arr + 1.).arr.compute()
```

Or if you have multiple files that you'd like to merge into a single array:
```py
open = lambda filename: xf.open(filename, chunks(None, None))
arr = xr.concat((map(open, files)), dim="files")
arr.mean()
```

Note indexing and chunking is currently far from optimal.
