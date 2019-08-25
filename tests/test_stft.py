import pytest
import numpy as np
import model
import test
import nnabla.functions as F
from nnabla.ext_utils import get_extension_context
import nnabla as nn

ctx = get_extension_context('cpu')
nn.set_default_context(ctx)


@pytest.fixture(params=[4096, 4096*10])
def nb_timesteps(request):
    return int(request.param)


@pytest.fixture(params=[1, 2, 3])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1])
def nb_samples(request):
    return request.param


@pytest.fixture(params=[1024, 2048, 4096])
def nfft(request):
    return int(request.param)


@pytest.fixture(params=[2, 4, 8])
def hop(request, nfft):
    return(nfft // request.param)


@pytest.fixture
def audio(request, nb_samples, nb_channels, nb_timesteps):
    return F.rand(shape=[nb_samples, nb_channels, nb_timesteps])


def test_stft(audio, nb_channels, nfft, hop):
    # clear STFT kernels (from previous tests with different frame size)
    nn.clear_parameters()

    # compute STFT using NNabla
    X_real, X_imag = model.STFT(audio, n_fft=nfft, n_hop=hop, center=True)
    nn.forward_all([X_real, X_imag])  # forward both at the same time to not create new random `audio`
    X = X_real.d + X_imag.d*1j

    # compute iSTFT using Scipy
    out = test.istft(X, n_fft=nfft, n_hopsize=hop)

    assert np.sqrt(np.mean((audio.d - out)**2)) < 1e-6
