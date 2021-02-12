# `xoundfile`

A Python library for loading sound files with `soundfile`, `xarray` and optionally `dask` support.

## Example

Lazy array:

```py
arr = open_soundfile(filename)
arr[:]
arr.compute()

```

Using Dask:

```py
arr = open_soundfile(filename, chunks=(10000, 2))
arr.compute()
```
