import random
from pathlib import Path
import numpy as np
import argparse
import musdb

from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain between `low` and `high`"""
    g = random.uniform(0.25, 1.25)
    return audio * g


def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and random.random() < 0.5:
        return np.flip(audio, 0)
    else:
        return audio


def load_datasources(parser, args, rng=None):
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """

    parser.add_argument('--is-wav', action='store_true', default=False,
                        help='loads wav instead of STEMS')
    parser.add_argument('--samples-per-track', type=int, default=64)
    parser.add_argument(
        '--source-augmentations', type=str, nargs='+',
        default=['gain', 'channelswap']
    )

    args = parser.parse_args()
    dataset_kwargs = {
        'root': args.root,
        'is_wav': args.is_wav,
        'subsets': 'train',
        'target': args.target,
        'download': args.root is None,
        'seed': args.seed
    }

    source_augmentations = Compose(
        [globals()['_augment_' + aug] for aug in args.source_augmentations]
    )

    train_dataset = MUSDBDataSource(
        split='train',
        samples_per_track=args.samples_per_track,
        seq_duration=args.seq_dur,
        source_augmentations=source_augmentations,
        random_track_mix=True,
        **dataset_kwargs
    )

    valid_dataset = MUSDBDataSource(
        split='valid', samples_per_track=1, seq_duration=args.seq_dur,
        **dataset_kwargs
    )

    return train_dataset, valid_dataset, args

class MUSDBDataSource(DataSource):
    def __init__(
        self,
        root=None,
        target='vocals',
        download=False,
        is_wav=False,
        subsets='train',
        split='train',
        seq_duration=6.0,
        samples_per_track=64,
        source_augmentations=lambda audio: audio,
        random_track_mix=False,
        dtype=np.float32,
        seed=42,
        rng=None,
        *args, **kwargs
    ):
        """
        see pytorch
        """
        super(MUSDBDataSource, self).__init__(shuffle=(split == 'train'))
        if rng is None:
            rng = np.random.RandomState(seed)
        self.rng = rng

        random.seed(seed)
        self.is_wav = is_wav
        self.seq_duration = seq_duration
        self.target = target
        self.subsets = subsets
        self.split = split
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix
        self.mus = musdb.DB(
            root=root,
            is_wav=is_wav,
            split=split,
            subsets=subsets,
            download=download,
            *args, **kwargs
        )
        self.sample_rate = 44100  # musdb is fixed sample rate
        self.dtype = dtype

        self._size = len(self.mus.tracks) * self.samples_per_track
        self._variables = ('mixture', 'target')
        self.reset()

    def _get_data(self, position):
        index = self._indexes[position]
        audio_sources = []
        target_ind = None

        # select track
        track = self.mus.tracks[index // self.samples_per_track]

        # at training time we assemble a custom mix
        if self.split == 'train' and self.seq_duration:
            for k, source in enumerate(self.mus.setup['sources']):
                # memorize index of target source
                if source == self.target:
                    target_ind = k

                # select a random track
                if self.random_track_mix:
                    track = random.choice(self.mus.tracks)

                # set the excerpt duration
                track.chunk_duration = self.seq_duration
                # set random start index
                track.chunk_start = random.uniform(
                    0, track.duration - self.seq_duration
                )
                # load source audio and apply time domain source_augmentations
                audio = track.sources[source].audio.T
                audio = self.source_augmentations(audio)
                audio_sources.append(audio)

            # create stem tensor of shape (source, channel, samples)
            stems = np.stack(audio_sources, axis=0)
            # # apply linear mix over source index=0
            x = np.sum(stems, axis=0)
            # get the target stem
            if target_ind is not None:
                y = stems[target_ind]
            # assuming vocal/accompaniment scenario if target!=source
            else:
                vocind = list(self.mus.setup['sources'].keys()).index('vocals')
                # apply time domain subtraction
                y = x - stems[vocind]

        # for validation and test, we deterministically yield the full
        # pre-mixed musdb track
        else:
            # get the non-linear source mix straight from musdb
            x = track.audio.T
            y = track.targets[self.target].audio.T

        return x, y

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(MUSDBDataSource, self).reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Open Unmix Dataset Test')

    parser.add_argument(
        '--root', type=str, help='root path of dataset'
    )

    parser.add_argument('--target', type=str, default='vocals')

    # I/O Parameters
    parser.add_argument(
        '--seq-dur', type=float, default=6.0,
        help='Duration of <=0.0 will result in the full audio'
    )

    parser.add_argument('--batch-size', type=int, default=16)

    args, _ = parser.parse_known_args()
    dataiter = data_iterator_musdb(root=args.root, target=args.target)
    for mixture, target in dataiter:
        print(mixture.shape)
        print(target.shape)
