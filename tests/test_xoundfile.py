import numpy as np
import soundfile as sf
import xoundfile as xf

import pytest


@pytest.fixture(params=[1, 4])
def nfiles(request):
    yield request.param


@pytest.fixture(
    params=[
        (lambda files: xf.open_files(files).compute()),  # xarray
        (
            lambda files: xf.open_files(files, chunks={"time": 10000, "channels": 2})
        ),  # dask
    ]
)
def function(request):
    yield request.param


def test_loading(tmpdir, function, nfiles):
    fs = 44100
    duration = 10
    nsamples = fs * duration
    nchannels = 2

    data = np.ones((nsamples, nchannels)) * 0.1
    sf.write(tmpdir / "file.wav", data, fs)
    files = [tmpdir / "file.wav"]
    files *= nfiles

    ds = function(files)
    arr = ds.signal

    assert len(arr.channel) == nchannels
    assert len(arr.time) == nsamples
    assert arr.shape == (nfiles, nsamples, nchannels)
    assert arr.channel.to_index().to_list() == [0, 1]
    # Check the values match. Note we want to select only a single filename to compare
    # hence the .isel
    np.testing.assert_array_almost_equal(arr.isel(filename=0).data, data, decimal=4)
