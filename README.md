#  _Open-Unmix_ for NNabla

[![Build Status](https://travis-ci.com/sigsep/open-unmix-nnabla.svg?branch=master)](https://travis-ci.com/sigsep/open-unmix-nnabla)

This repository contains the NNabla implementation of __Open-Unmix__, a deep neural network reference implementation for music source separation, applicable for researchers, audio engineers and artists. __Open-Unmix__ provides ready-to-use models that allow users to separate pop music into four stems: __vocals__, __drums__, __bass__ and the remaining __other__ instruments.

⚠️ Please note that this implementation was not tested to yield the same results as the pytorch version. For that reason, we do not provide an inference method in the nnabla version for now. If you want to use `umx` or `umx-hq` in your work, use the pytorch version for now. See the current implementation status [here](https://github.com/sigsep/open-unmix-nnabla/issues/1)

__Related Projects:__ [open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch) | open-unmix-nnabla | [open-unmix-tensorflow](https://github.com/sigsep/open-unmix-tensorflow) | [musdb](https://github.com/sigsep/sigsep-mus-db) | [museval](https://github.com/sigsep/sigsep-mus-eval) | [norbert](https://github.com/sigsep/norbert)

## The Model

![](https://docs.google.com/drawings/d/e/2PACX-1vTPoQiPwmdfET4pZhue1RvG7oEUJz7eUeQvCu6vzYeKRwHl6by4RRTnphImSKM0k5KXw9rZ1iIFnpGW/pub?w=959&h=308)

_Open-Unmix_ is based on a three-layer bidirectional deep LSTM. The model learns to predict the magnitude spectrogram of a target, like _vocals_, from the magnitude spectrogram of a mixture input. Internally, the prediction is obtained by applying a mask on the input. The model is optimized in the magnitude domain using mean squared error and the actual separation is done in a post-processing step involving a multichannel wiener filter implemented using [norbert](https://github.com/sigsep/norbert). To perform separation into multiple sources, multiple models are trained for each particular target. While this makes the training less comfortable, it allows great flexibility to customize the training data for each target source.

### Input Stage

__Open-Unmix__ operates in the time-frequency domain to perform its prediction. The input of the model is either:

* __A time domain__ signal tensor of shape `(nb_samples, nb_channels, nb_timesteps)`, where `nb_samples` are the samples in a batch, `nb_channels` is 1 or 2 for mono or stereo audio, respectively, and `nb_timesteps` is the number of audio samples in the recording.

 In that case, the model computes spectrograms with nnabla `STFT` on the fly.

* Alternatively _open-unmix_ also takes **magnitude spectrograms** directly (e.g. when pre-computed and loaded from disk).

 In that case, the input is of shape `(nb_frames, nb_samples, nb_channels, nb_bins)`, where `nb_frames` and `nb_bins` are the time and frequency-dimensions of a Short-Time-Fourier-Transform.

The input spectrogram is _standardized_ using the global mean and standard deviation for every frequency bin across all frames. Furthermore, we apply batch normalization in multiple stages of the model to make the training more robust against gain variation.

### Dimensionality reduction

The LSTM is not operating on the original input spectrogram resolution. Instead, in the first step after the normalization, the network learns to compresses the frequency and channel axis of the model to reduce redundancy and make the model converge faster.

### Bidirectional-LSTM

The core of __open-unmix__ is a three layer bidirectional [LSTM network](https://dl.acm.org/citation.cfm?id=1246450). Due to its recurrent nature, the model can be trained and evaluated on arbitrary length of audio signals. Since the model takes information from past and future simultaneously, the model cannot be used in an online/real-time manner.
An uni-directional model can easily be trained as described [here](docs/training.md).

### Output Stage

After applying the LSTM, the signal is decoded back to its original input dimensionality. In the last steps the output is multiplied with the input magnitude spectrogram, so that the models is asked to learn a mask.

## Separation

Inference is not supported right now and will be added later.

## Getting started

### Installation

For installation we recommend to use the [Anaconda](https://anaconda.org/) python distribution. To create a conda environment for _open-unmix_, simply run:

`conda env create -f environment-X.yml` where `X` is either [`cpu`, `gpu`], depending on your system. 

## Design Choices / Contributions

* we favored simplicity over performance to promote clearness of the code. The rationale is to have __open-unmix__ serve as a __baseline__ for future research while performance still meets current state-of-the-art (See [Evaluation](#Evaluation)). The results are comparable/better to those of `UHL1`/`UHL2` which obtained the best performance over all systems trained on MUSDB18 in the [SiSEC 2018 Evaluation campaign](https://sisec18.unmix.app).
* We designed the code to allow researchers to reproduce existing results, quickly develop new architectures and add own user data for training and testing. We favored framework specifics implementations instead of having a monolithic repository.
* _open-unmix_ is a community focused project, we therefore encourage the community to submit bug-fixes and comments and improve the computational performance. However, we are not looking for changes that only focused on improving the performance.

### Authors

[Fabian-Robert Stöter](https://www.faroit.com/), [Antoine Liutkus](https://github.com/aliutkus), Inria and LIRMM, Montpellier, France

## References

<details><summary>If you use open-unmix for your research – Cite Open-Unmix</summary>
  
```latex
@article{stoter19,  
  author={F.-R. St\\"oter and S. Uhlich and A. Liutkus and Y. Mitsufuji},  
  title={Open-unmix: a reference implementation for source separation},  
  journal={Journal of Open Source Software},  
  year=2019,  
  note={submitted}}
}
```

</p>
</details>

<details><summary>If you use the MUSDB dataset for your research - Cite the MUSDB18 Dataset</summary>
<p>

```latex
@misc{MUSDB18,
  author       = {Rafii, Zafar and
                  Liutkus, Antoine and
                  Fabian-Robert St{\"o}ter and
                  Mimilakis, Stylianos Ioannis and
                  Bittner, Rachel},
  title        = {The {MUSDB18} corpus for music separation},
  month        = dec,
  year         = 2017,
  doi          = {10.5281/zenodo.1117372},
  url          = {https://doi.org/10.5281/zenodo.1117372}
}
```

</p>
</details>


<details><summary>If compare your results with SiSEC 2018 Participants - Cite the SiSEC 2018 LVA/ICA Paper</summary>
<p>

```latex
@inproceedings{SiSEC18,
  author="St{\"o}ter, Fabian-Robert and Liutkus, Antoine and Ito, Nobutaka",
  title="The 2018 Signal Separation Evaluation Campaign",
  booktitle="Latent Variable Analysis and Signal Separation:
  14th International Conference, LVA/ICA 2018, Surrey, UK",
  year="2018",
  pages="293--305"
}
```

</p>
</details>

⚠️ Please note that the official acronym for _open-unmix_ is **UMX**.

### License

MIT
