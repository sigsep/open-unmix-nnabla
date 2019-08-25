import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.logger as logger
import numpy as np


def STFT(x, n_fft=4096, n_hop=1024, center=False):
    """Multichannel STFT
    
    Input: (nb_samples, nb_channels, nb_timesteps)
    Output: (nb_samples, nb_channels, nb_bins, nb_frames), 
            (nb_samples, nb_channels, nb_bins, nb_frames)
    """
    nb_samples, nb_channels, nb_timesteps = x.shape
    x = x.reshape((nb_samples*nb_channels, -1))

    real, imag = F.stft(
        x, n_fft, n_hop, n_fft,
        window_type='hanning',
        center=center,
        pad_mode='reflect'
    )

    real = real.reshape(
        (nb_samples, nb_channels, n_fft // 2 + 1, -1)
    )

    imag = imag.reshape(
        (nb_samples, nb_channels, n_fft // 2 + 1, -1)
    )

    return real, imag


def Spectrogram(real, imag, power=1, mono=True):
    """
    Input:  (nb_samples, nb_channels, nb_bins, nb_frames), 
            (nb_samples, nb_channels, nb_bins, nb_frames)
    Output: (nb_frames, nb_samples, nb_channels, nb_bins)
    """
    spec = ((real ** 2) + (imag ** 2)) ** (power / 2.0)

    if mono:
        spec = F.mean(spec, axis=1, keepdims=True)

    return F.transpose(spec, ((3, 0, 1, 2)))


class OpenUnmix(object):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        hidden_size=512,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        unidirectional=False,
        power=1
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
               (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        super(OpenUnmix, self).__init__()

        self.nb_output_bins = n_fft // 2 + 1

        self.hidden_size = hidden_size
        self.input_is_spectrogram = input_is_spectrogram
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.power = power
        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.unidirectional = unidirectional

        if unidirectional:
            self.nb_of_directions = 1
        else:
            self.nb_of_directions = 2

        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        if input_mean is None:
                input_mean = np.zeros((self.nb_bins,))
        self.input_mean = nn.parameter.get_parameter_or_create(
            'input_mean',
            shape=None,
            initializer=-input_mean[:self.nb_bins],
            need_grad=True
        )

        if input_scale is None:
            input_scale = np.ones((self.nb_bins))
        self.input_scale = nn.parameter.get_parameter_or_create(
            'input_scale',
            shape=None,
            initializer=1.0/input_scale[:self.nb_bins],
            need_grad=True
        )

        self.output_scale = nn.parameter.get_parameter_or_create(
            'output_scale',
            shape=None,
            initializer=np.ones((self.nb_output_bins,)),
            need_grad=True
        )

        self.output_mean = nn.parameter.get_parameter_or_create(
            'output_mean',
            shape=None,
            initializer=np.ones((self.nb_output_bins,)),
            need_grad=True
        )

    def __call__(self, x, test=False):
        
        # x = PF.mean_subtraction(x, base_axis=0)
        if not self.input_is_spectrogram:
            x = Spectrogram(
                *STFT(x, n_fft=self.n_fft, n_hop=self.n_hop),
                power=self.power, mono=(self.nb_channels == 1)
            )

        nb_frames, nb_samples, nb_channels, nb_bins = x.shape

        mix = x

        x = x[..., :self.nb_bins]
        x += F.reshape(self.input_mean, shape=(1, 1, 1, self.nb_bins), inplace=False)
        x *= F.reshape(self.input_scale, shape=(1, 1, 1, self.nb_bins), inplace=False)

        with nn.parameter_scope("fc1"):
            x = PF.affine(x, self.hidden_size, base_axis=2)
            x = PF.batch_normalization(x, batch_stat=not test)
            x = F.tanh(x)

        with nn.parameter_scope("lstm"):
            if self.unidirectional:
                lstm_hidden_size = self.hidden_size
            else:
                lstm_hidden_size = self.hidden_size // 2

            h = nn.Variable(
                (self.nb_layers, self.nb_of_directions, nb_samples, lstm_hidden_size),
                need_grad=False
            )
            h.d = np.zeros(h.shape)
            c = nn.Variable(
                (self.nb_layers, self.nb_of_directions, nb_samples, lstm_hidden_size),
                need_grad=False
            )
            c.d = np.zeros(c.shape)
            lstm_out, _, _ = PF.lstm(x, h, c, num_layers=self.nb_layers, bidirectional=not self.unidirectional, training=not test)

        x = F.concatenate(x, lstm_out)  # concatenate along last axis

        with nn.parameter_scope("fc2"):
            x = PF.affine(
                x, (self.hidden_size), base_axis=2,
            )
            x = PF.batch_normalization(x, batch_stat=not test)
            x = F.relu(x)

        with nn.parameter_scope("fc3"):
            x = PF.affine(
                x, (nb_channels, nb_bins), base_axis=2,
            )
            x = PF.batch_normalization(x, batch_stat=not test)

        x = x.reshape((nb_frames, nb_samples, nb_channels, self.nb_output_bins))

        # apply output scaling
        x *= F.reshape(self.output_scale, shape=(1, 1, 1, self.nb_output_bins), inplace=False)
        x += F.reshape(self.output_mean, shape=(1, 1, 1, self.nb_output_bins), inplace=False)
        x = F.relu(x) * mix

        return x


if __name__ == "__main__":
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context('cudnn')
    nn.set_default_context(ctx)

    x = np.random.randn(255, 16, 2, 2049)
    nx = nn.Variable.from_numpy_array(x)
    print(nx.shape)
    unmix = OpenUnmix(input_is_spectrogram=True)

    # create model
    X = unmix(nx)
    print(X.shape)

    # perform forward pass
    X.forward()
    X.backward()
    print('finished')

