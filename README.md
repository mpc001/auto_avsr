# Auto-AVSR: Lip-Reading Sentences Project

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/auto-avsr-audio-visual-speech-recognition/lipreading-on-lrs3-ted)](https://paperswithcode.com/sota/lipreading-on-lrs3-ted?p=auto-avsr-audio-visual-speech-recognition)

## Update

`2023-07-26`: We released the implementation of [Real-Time AV-ASR](https://github.com/pytorch/audio/tree/main/examples/avsr).

## Introduction

This repository is an open-sourced framework for speech recognition, with a primary focus on visual speech (lip-reading). It is designed for end-to-end training, aiming to deliver state-of-the-art models and enable reproducibility on audio-visual speech benchmarks.

<div align="center"><img src="doc/pipeline.png" width="640"/></div>

By using this repository, you can achieve a word error rate (WER) of 20.3% for visual speech recognition (VSR) and 1.0% for audio speech recognition (ASR) on LRS3.

## Setup

1. Set up environment:

```Shell
conda create -y -n auto_avsr python=3.8
conda activate auto_avsr
```

2. Clone repository:

```Shell
git clone https://github.com/mpc001/auto_avsr
cd auto_avsr
```

3. Install fairseq within the repository:

```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
```

4. Install PyTorch (tested pytorch version: v2.0.1) and other packages:

```Shell
pip install torch torchvision torchaudio
pip install pytorch-lightning==1.5.10
pip install sentencepiece
pip install av
pip install hydra-core --upgrade
```

5. Install ffmpeg:

```
conda install "ffmpeg<5" -c conda-forge
```

6. Prepare the dataset. See the instructions in the [preparation](./preparation) folder.

## Training

```Shell
python train.py exp_dir=[exp_dir] \
               exp_name=[exp_name] \
               data.modality=[modality] \
               data.dataset.root_dir=[root_dir] \
               data.dataset.train_file=[train_file] \
               trainer.num_nodes=[num_nodes] \
```
<details open>
  <summary><strong>Required arguments</strong></summary>

- `exp_dir`: Directory to save checkpoints and logs to.
- `exp_name`: Experiment name. Location of checkpoints is `[exp_dir]`/`[exp_name]`.
- `data.modality`: Type of input modality, valid values: `video` and `audio`.
- `data.dataset.root_dir`: Root directory of preprocessed dataset, default: `null`.
- `data.dataset.train_file`: Filename of training label list, default: `lrs3_train_transcript_lengths_seg24s.csv`.
- `trainer.num_nodes`: Number of machines used, default: 1.
- `trainer.resume_from_checkpoint`: Path of the checkpoint from which training is resumed, default: `null`.

</details>

<details>
  <summary><strong>Optional arguments</strong></summary>

- `data.dataset.val_file`: Filename of validation label list, default: `lrs3_test_transcript_lengths_seg24s.csv`.
- `pretrained_model_path`: Path to the pre-trained model, default: `null`.
- `transfer_frontend` Flag to load the weights of front-end module, works with `pretrained_model_path`.
- `transfer_encoder` Flag to load the weights of encoder, works with `pretrained_model_path`.
- `trainer.max_epochs`: Number of epochs, default: 75.
- `trainer.gpus`: Number of GPUs to train on on each machine, default: -1, which use all gpus.
- `data.max_frames`: Maximal number of frames in a batch, default: 1800.
- `optimizer.lr`: Learning rate, default: 0.001.

</details>


<details open>
  <summary><strong>Note</strong></summary>

- For lrs3, start by training from scratch on a subset (23h, max duration=4 seconds) at a learning rate of 0.0002 (see [model-zoo](#model-zoo)). Then fine-tune on the full set with a learning rate of 0.001. A script for subset creation is available [here](./preparation/limit_length.py). For training new datasets, please refer to [instruction](INSTRUCTION.md).
- If you want to monitor the training process, customise [logger](https://lightning.ai/docs/pytorch/1.5.8/api_references.html#loggers-api) within `pytorch_lightning.Trainer()`.
- To maximize resource utilization, set `data.max_frames` to the largest to fit into your GPU memory.

</details>

## Testing

```Shell
python eval.py data.modality=[modality] \
               data.dataset.root_dir=[root_dir] \
               data.dataset.test_file=[test_file] \
               pretrained_model_path=[pretrained_model_path] \
```

<details open>
  <summary><strong>Required arguments</strong></summary>

- `data.modality`: Type of input modality, valid values: `video`, `audio` and `audiovisual`.
- `data.dataset.root_dir`: Root directory of preprocessed dataset, default: `null`.
- `data.dataset.test_file`: Filename of testing label list, default: `lrs3_test_transcript_lengths_seg24s.csv`.
- `pretrained_model_path`: Path to the pre-trained model, set to `[exp_dir]/[exp_name]/model_avg_10.pth`, default: `null`.

</details>

<details>
  <summary><strong>Optional arguments</strong></summary>

- `decode.snr_target=[snr_target]`: Level of signal-to-noise ratio (SNR), default: 999999.

</details>

## Demo

Want to see how our asr/vsr model performs on your audio/video? Just run this command:

```Shell
python demo.py  data.modality=[modality] \
                pretrained_model_path=[pretrained_model_path] \
                file_path=[file_path]
```
<details open>
  <summary><strong>Required arguments</strong></summary>

- `data.modality`: Type of input modality, valid values: `video` and `audio`.
- `pretrained_model_path`: Path to the pre-trained model.
- `file_path`: Path to the file for testing.

</details>


## Model zoo

We provide audio-only, visual-only and audio-visual models for lrs3.

<details open>

<summary>LRS3</summary>

| Model                                 | Training data (h)  |  WER [%]   |    MD5            |
|---------------------------------------|:------------------:|:----------:|:------------------------:|
| [`vsr_trlrs3_23h_base.pth`](https://drive.google.com/file/d/1OBEHbStKKFG7VDij14RDLN9VYSdE_Bhs/view?usp=sharing)             |        23           |    96.6    | 50c88  |
| [`vsr_trlrs3_base.pth`](https://drive.google.com/file/d/1aawSjxIL2ewo0W0fg4TBQgR8WMAmPeSL/view?usp=sharing)                 |        438          |    36.7    | ea3ec  |
| [`vsr_trlrs3vox2_base.pth`](https://drive.google.com/file/d/1mLAuCnK2y7zbmiHlAXMqPSF_ApGqfbAD/view?usp=sharing)             |        1759         |    25.0    | 0a126  |
| [`vsr_trlrwlrs2lrs3vox2avsp_base.pth`](https://drive.google.com/file/d/19GA5SqDjAkI5S88Jt5neJRG-q5RUi5wi/view?usp=sharing)  |        3448         |    20.3    | a896f  |
| [`asr_trlrs3_23h_base.pth`](https://drive.google.com/file/d/1ERXLjBGFQDAXXKkHLBrVi6xI7l1QiKyL/view?usp=sharing)             |        23           |    72.5    | 87d45  |
| [`asr_trlrs3_base.pth`](https://drive.google.com/file/d/1FuYLkBt6DFzxIR7AbCs6jzhbLfaJMk6a/view?usp=sharing)                 |        438          |    2.04    | 4fa87  |
| [`asr_trlrs3vox2_base.pth`](https://drive.google.com/file/d/13o_KvPeLHkjFPVm28Gvn8EQNBkS5ZBV6/view?usp=sharing)             |        1759         |    1.07    | 7beab  |
| [`asr_trlrwlrs2lrs3vox2avsp_base.pth`](https://drive.google.com/file/d/12vigJjL_ipgRz5CMYYQPdn8edEXD-Cuq/view?usp=sharing)  |        3448         |    0.99    | dc759  |
| [`avsr_trlrwlrs2lrs3vox2avsp_base.pth`](https://drive.google.com/file/d/1mU6MHzXMiq1m6GI-8gqT2zc2bdStuBXu/view?usp=sharing)  |        3448         |    0.93    | 6b3c5  |

</details>

## Citation

If you find this repository helpful, please consider citing our work:

```bibtex
@inproceedings{ma2023auto,
  author={Ma, Pingchuan and Haliassos, Alexandros and Fernandez-Lopez, Adriana and Chen, Honglie and Petridis, Stavros and Pantic, Maja},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels},
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096889}
}
```

## Acknowledgement

This repository is built using the [espnet](https://github.com/espnet/espnet), [fairseq](https://github.com/facebookresearch/fairseq), [raven](https://github.com/ahaliassos/raven) and [avhubert](https://github.com/facebookresearch/av_hubert) repositories.

## License

Code is Apache 2.0 licensed. The pre-trained models provided in this repository may have their own licenses or terms and conditions derived from the dataset used for training.

## Contact

Contributions are welcome; feel free to create a PR or email me:

```
[Pingchuan Ma](pingchuan.ma16[at]imperial.ac.uk)
```
