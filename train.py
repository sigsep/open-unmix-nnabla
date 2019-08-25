import os
import argparse
import model
import data
import nnabla.utils.save as save
import nnabla.solvers as S
import nnabla.parametric_functions as PF
import nnabla.functions as F
import nnabla.logger as logger
import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_iterator import data_iterator
import numpy as np
from numpy.random import RandomState
from numpy.random import seed
import sklearn.preprocessing
import tqdm
import copy
import utils
seed(42)


def get_args():
    parser = argparse.ArgumentParser(description='Open Unmix Trainer')

    # which target do we want to train?
    parser.add_argument('--target', type=str, default='vocals',
                        help='target source (will be passed to the dataset)')

    # Dataset paramaters
    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--output', type=str, default="open-unmix",
                        help='provide output path base folder name')

    # Trainig Parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    parser.add_argument('--patience', type=int, default=140,
                        help='maximum number of epochs to train (default: 140)')
    parser.add_argument('--lr-decay-patience', type=int, default=80,
                        help='lr decay patience for plateau scheduler')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.3,
                        help='gamma of learning rate scheduler decay')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    # Model Parameters
    parser.add_argument('--seq-dur', type=float, default=6.0,
                        help='Sequence duration in seconds'
                        'value of <=0.0 will use full/variable length')
    parser.add_argument('--unidirectional', action='store_true', default=False,
                        help='Use unidirectional LSTM instead of bidirectional')
    parser.add_argument('--nfft', type=int, default=4096,
                        help='STFT fft size and window size')
    parser.add_argument('--nhop', type=int, default=1024,
                        help='STFT hop size')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='hidden size parameter of dense bottleneck layers')
    parser.add_argument('--bandwidth', type=int, default=16000,
                        help='maximum model bandwidth in herz')
    parser.add_argument('--nb-channels', type=int, default=2,
                        help='set number of channels for model (1, 2)')
    parser.add_argument('--nb-workers', type=int, default=0,
                        help='Number of workers for dataloader.')

    # Misc Parameters
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='less verbose during training')

    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--model-save-interval", "-s", type=int, default=1000,
                        help='The interval of saving model parameters.')
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension modules. ex) 'cpu', 'cudnn'.")

    args, _ = parser.parse_known_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    return parser, args


def get_statistics(args, datasource):
    scaler = sklearn.preprocessing.StandardScaler()

    pbar = tqdm.tqdm(range(len(datasource.mus.tracks)), disable=args.quiet)

    for ind in pbar:
        x = datasource.mus.tracks[ind].audio.T
        audio = nn.Variable([1] + list(x.shape))
        audio.d = x

        target_spec = model.Spectrogram(
            *model.STFT(audio, n_fft=args.nfft, n_hop=args.nhop),
            mono=(args.nb_channels == 1)
        )

        pbar.set_description("Compute dataset statistics")
        target_spec.forward()
        scaler.partial_fit(np.squeeze(target_spec.d[0]))
    # set inital input scaler values
    std = np.maximum(
        scaler.scale_,
        1e-4*np.max(scaler.scale_)
    )
    return scaler.mean_, std


def train():
    parser, args = get_args()

    # Get context.
    ctx = get_extension_context(args.context, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Initialize DataIterator for MNIST.
    train_source, valid_source, args = data.load_datasources(
        parser, args, rng=RandomState(42)
    )

    train_iter = data_iterator(
        train_source,
        args.batch_size,
        RandomState(args.seed),
        with_memory_cache=False,
        with_file_cache=False
    )

    valid_iter = data_iterator(
        valid_source,
        args.batch_size,
        RandomState(args.seed),
        with_memory_cache=False,
        with_file_cache=False
    )

    scaler_mean, scaler_std = get_statistics(args, train_source)

    max_bin = utils.bandwidth_to_max_bin(
        train_source.sample_rate, args.nfft, args.bandwidth
    )

    unmix = model.OpenUnmix(
        input_mean=scaler_mean,
        input_scale=scaler_std,
        nb_channels=args.nb_channels,
        hidden_size=args.hidden_size,
        n_fft=args.nfft,
        n_hop=args.nhop,
        max_bin=max_bin,
        sample_rate=train_source.sample_rate
    )

    # Create input variables.
    audio_shape = [args.batch_size] + list(train_source._get_data(0)[0].shape)
    mixture_audio = nn.Variable(audio_shape)
    target_audio = nn.Variable(audio_shape)

    vmixture_audio = nn.Variable(audio_shape)
    vtarget_audio = nn.Variable(audio_shape)

    # create train graph
    pred_spec = unmix(mixture_audio, test=False)
    pred_spec.persistent = True

    target_spec = model.Spectrogram(
        *model.STFT(target_audio, n_fft=unmix.n_fft, n_hop=unmix.n_hop),
        mono=(unmix.nb_channels == 1)
    )

    loss = F.mean(F.squared_error(pred_spec, target_spec), axis=1)

    # Create Solver.
    solver = S.Adam(args.lr)
    solver.set_parameters(nn.get_parameters())

    # Training loop.
    t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
    es = utils.EarlyStopping(patience=args.patience)

    for epoch in t:
        # TRAINING
        t.set_description("Training Epoch")
        b = tqdm.trange(0, train_source._size // args.batch_size, disable=args.quiet)
        losses = utils.AverageMeter()
        for batch in b:
            mixture_audio.d, target_audio.d = train_iter.next()
            b.set_description("Training Batch")
            solver.zero_grad()
            loss.forward(clear_no_need_grad=True)
            loss.backward(clear_buffer=True)
            solver.weight_decay(args.weight_decay)
            solver.update()
            losses.update(loss.d.copy().mean())
            b.set_postfix(
                train_loss=losses.avg
            )

        # VALIDATION
        vlosses = utils.AverageMeter()
        for batch in range(valid_source._size):
            # Create new validation input variables for every batch
            vmixture_audio.d, vtarget_audio.d = valid_iter.next()
            # create validation graph
            vpred_spec = unmix(vmixture_audio, test=True)
            vpred_spec.persistent = True

            vtarget_spec = model.Spectrogram(
                *model.STFT(vtarget_audio, n_fft=unmix.n_fft, n_hop=unmix.n_hop),
                mono=(unmix.nb_channels == 1)
            )
            vloss = F.mean(F.squared_error(vpred_spec, vtarget_spec), axis=1)

            vloss.forward(clear_buffer=True)
            vlosses.update(vloss.d.copy().mean())

        t.set_postfix(
            train_loss=losses.avg, val_loss=vlosses.avg
        )

        stop = es.step(vlosses.avg)
        is_best = vlosses.avg == es.best

        # save current model
        nn.save_parameters(os.path.join(
            args.output, 'checkpoint_%s.h5' % args.target))

        if is_best:
            best_epoch = epoch
            nn.save_parameters(os.path.join(
                args.output, '%s.h5' % args.target))

        if stop:
            print("Apply Early Stopping")
            break


if __name__ == '__main__':
    train()
