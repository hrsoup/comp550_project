# Sentence Segmentation via Studying Knowledge from Symbolic Music

## Installation

We use a single GPU (NVIDIA GeForce MX250) to develop this system with:
- Anaconda 3 (python 3.7)
- torch 1.10.0+cu113
- sklearn
- nltk 3.6.1

GPU is optinal for running this system.

If you have installed Anaconda 3, you can use the following bash code to create conda environment and install all the packages needed in this system:

```bash
conda create -n sentence_segmentation python=3.7
conda activate sentence_segmentation
pip install sklearn
pip install nltk
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
% (if not gpu) conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## Usage

(1) Using shuffled music data to pretrain, language data to finetune.

```bash
python main.py RandomMusic_Language
```

(2) Using correct music data to pretrain, language data to finetune.
```bash
python main.py Music_Language
```

(3) Only using language data to train.

```bash
python main.py Language
```

## Results

The original experimental results of our paper are in the `results` folder.
