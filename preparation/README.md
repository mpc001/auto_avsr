
# Preprocessing

We provide a pre-processing pipeline in this repository for detecting and cropping mouth regions of interest (ROIs) as well as corresponding audio waveforms for LRS2, LRS3, and VoxCeleb2.

## Prerequisites

1. Install all dependency-packages.

```Shell
pip install -r requirements.txt
```

2. Install [retinaface](./tools) or [mediapipe](https://pypi.org/project/mediapipe/) tracker. If you have installed the tracker, please skip it.

## Preprocessing LRS2 or LRS3

To pre-process the LRS2 or LRS3 dataset, plrase follow these steps:

1. Download the LRS2 or LRS3 dataset from the official website.

2. If you leave `landmarks-dir` empty, the face detector will automatically track landmarks. However, if you prefer to use our pre-tracked landmarks from retinaface tracker, you can download them below.

| File Name              | Source URL                                                                              | File Size  |
|------------------------|-----------------------------------------------------------------------------------------|------------|
| LRS3_landmarks.zip     |[GoogleDrive](https://bit.ly/33rEsax) or [BaiduDrive](https://bit.ly/3rwQSph)(key: mi3c) |     18GB   |
| LRS2_landmarks.zip     |[GoogleDrive](https://bit.ly/3jSMMoz) or [BaiduDrive](https://bit.ly/3BuIwBB)(key: 53rc) |     9GB    |

3. Run the following command to preprocess the dataset:

```Shell
python preprocess_lrs2lrs3.py \
    --data-dir [data_dir] \
    --landmarks-dir [landmarks_dir] \
    --detector [detector] \
    --root-dir [root_dir] \
    --dataset [dataset] \
    --subset [subset] \
    --seg-duration [seg_duration] \
    --groups [n] \
    --job-index [j]
```
- `data-dir`: Path to the directory containing video files.
- `landmarks-dir`: Path to the directory containing landmarks files. If the `landmarks-dir` is specified, face detector will not be used.
- `detector`: Type of face detector. Valid values are: `mediapipe` and `retinaface`. Default: `retinaface`.
- `root-dir`: Path to the root directory where all preprocessed files will be stored.
- `dataset`: Name of the dataset. Valid values are: `lrs2` and `lrs3`.
- `subset`: For `lrs2`, the subset can be `train`, `val`, and `test`. For `lrs3`, the subset can be `train` and `test`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `24`.
- `groups`: Number of groups to split the dataset into.
- `job-index`: Job index for the current group. Valid values are an integer within the range of `[0, n)`.

3. Run the following command to merge all labels:

```Shell
python merge.py \
    --root-dir [root_dir] \
    --dataset [dataset] \
    --subset [subset] \
    --seg-duration [seg_duration] \
    --groups [n]
```
- `root-dir`: Path to the root directory where all preprocessed files will be stored.
- `dataset`: Name of the dataset. Valid values are: `lrs2` and `lrs3`.
- `subset`: The subset name of the dataset. For LRS2, valid values are `train`, `val`, and `test`. For LRS3, valid values are `train` and `test`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `24`.
- `groups`: Number of groups to split the dataset into.

## VoxCeleb2 Preprocessing
To pre-process the VoxCeleb2 dataset, please follow these steps:

1. Download the VoxCeleb2 dataset from the official website.

2. If you leave the `landmarks-dir` argument empty, our face detector will automatically track the landmarks. However, if you'd prefer to use our pre-tracked landmarks from the retinaface tracker, you can download them below. Once you've finished downloading the five files, simply merge them into one single file using `zip -FF vox2_landmarks.zip --out single.zip`, and then decompress it.

| File Name              | Source URL                                                                        | File Size |
|------------------------|-----------------------------------------------------------------------------------|-----------|
| vox2_landmarks.zip     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.zip)     | 18GB      |
| vox2_landmarks.z01     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z01)     | 20GB      |
| vox2_landmarks.z02     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z02)     | 20GB      |
| vox2_landmarks.z03     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z03)     | 20GB      |
| vox2_landmarks.z04     | [Download](https://www.doc.ic.ac.uk/~pm4115/vox2landmarks/vox2_landmarks.z04)     | 20GB      |

3. Run the following command to preprocess the dataset:

```Shell
python preprocess_vox2.py \
    --vid-dir [vid_dir] \
    --aud-dir [aud_dir] \
    --label-dir [label_dir] \
    --landmarks-dir [landmarks_dir] \
    --detector [detector] \
    --root-dir [root_dir] \
    --dataset [dataset] \
    --seg-duration [seg_duration] \
    --groups [n] \
    --job-index [j]
```
- `vid-dir`: Path to the directory containing video files.
- `aud-dir`: Path to the directory containing audio files.
- `label-dir`: Path to the directory containing language-identification label files. Default: ``. For the label file, we use `vox-en.id` provided by [AVHuBERT repository](https://github.com/facebookresearch/av_hubert/tree/5ab235b3d9dac548055670d534b283b5b70212cc/avhubert/preparation/data).
- `landmarks-dir`: Path to the directory containing landmarks files. If the `landmarks-dir` is specified, face detector will not be used.
- `detector`: Type of face detector. Valid values are: `mediapipe` and `retinaface`. Default: `retinaface`.
- `root-dir`: Path to the root directory where all preprocessed files will be stored.
- `dataset`: Name of the dataset. Default: `vox2`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `24`.
- `groups`: Number of groups to split the dataset into.
- `job-index`: Job index for the current group and should be an integer within the range of `[0, n)`.

This command will preprocess the dataset and store the preprocessed files in the specified `[root_dir]`/`[dataset]`.

4. Install the pre-trained ASR model, such as [whisper](https://github.com/openai/whisper).

5. Run the following command to generate transcripts:

```Shell
python asr_infer.py \
    --root-dir [root-dir] \
    --dataset [dataset] \
    --seg-duration [seg_duration] \
    --groups [n] \
    --job-index [j]
```
- `root-dir`: Path to the root directory where all preprocessed files are stored.
- `dataset`: Name of the dataset. Valid value is: `vox2`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `24`.
- `groups`: Number of groups the dataset was split into during preprocessing.
- `job-index`: Job index for the current group.

6. Run the following command to merge all labels. (Same as the merge solution at [preprocessing-lrs2-or-lrs3](#preprocessing-lrs2-or-lrs3))

```Shell
python merge.py \
    --root-dir [root_dir] \
    --dataset [dataset] \
    --subset [subset] \
    --seg-duration [seg_duration] \
    --groups [n]
```
- `root-dir`: Path to the root directory where all preprocessed files will be stored.
- `dataset`: Name of the dataset. Valid value is: `vox2`
- `subset`: The subset name of the dataset. For `vox2`, valid value is `train`.
- `seg-duration`: Length of the maximal segment in seconds. Default: `24`.
- `groups`: Number of groups to split the dataset into.
