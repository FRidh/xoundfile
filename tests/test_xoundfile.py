import numpy as np
import soundfile as sf
import xoundfile as xf

import pytest


@pytest.mark.parametrize(
    "function",
    [
        (xf.open_soundfile),  # xarray
        (lambda filename: xf.open_soundfile(filename, chunks=(10000, 2))),  # dask
    ],
)
def test_loading(tmpdir, function):
    fs = 44100
    duration = 10
    nsamples = fs * duration
    nchannels = 2

    data = np.ones((nsamples, nchannels)) * 0.1
    sf.write(tmpdir / "file.wav", data, fs)
    arr = function(tmpdir / "file.wav")
    assert len(arr.channel) == nchannels
    assert len(arr.time) == nsamples
    assert arr.shape == (nsamples, nchannels)
    np.testing.assert_array_almost_equal(arr.data, data, decimal=4)
