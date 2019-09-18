# Attention-based-CNN-for-sentence-classification
# Getting Started
This is the code for the paper https://www.isca-speech.org/archive/Interspeech_2016/pdfs/0354.PDF. This is one of the few papers that applies attention on word embeddings.

# Requirements

```
python>=3.5

pytorch>=1.0.0

numpy

sklearn

```
# How to run

## Step:1

Download glove 840B vectors https://nlp.stanford.edu/projects/glove/ . Unzip it and mention the file path in the variable **vecpath** in **vector.py** in the PyTorch Folder.

## Step:2

```
git clone https://github.com/avinashsai/Attention-based-CNN-for-sentence-classification.git

cd PyTorch

```

## Step:3

```
python main.py --dataset  --runs  --lamda 

dataset [Training Dataset] options (trec/mr/sst5/sst2 : default - trec)

runs [Number of average runs to report results] options ( 5/10 : default - 10)

lamda [ Distance decay (λ) ] options ( [0,1) : default - 0.0 ; For TREC Best lamda is 0.02 )

```
# Results

For TREC Dataset:

| Distance decay (λ) | This Implementation Result | Paper Result | 
|--------------------|----------------------------|--------------|
| 0.00 | 94.60  | **95.20** | 
| 0.01 | 96.24  |  -    |
| 0.02 | **96.48**  | 95.00 |    
| 0.03 | 95.90  |  -    |
| 0.04 | 96.12  | 94.80 |
| 0.05 | 95.52  | -     |
| 0.06 | 95.76  | 95.20 |
| 0.07 | 95.72  |  -    |
| 0.08 | 94.88  | 94.40 |
| 0.09 | 95.76  | -    |
| 0.10 | 95.94  | 94.60|
| 0.11 | 96.36  | -    |
| 0.12 | 95.61  | 94.60|
| 0.13 | 95.80  | -    |
| 0.14 | 95.56  | 94.20|
| 0.15 | 95.76  | -    |
| 0.16 | 95.58  | 94.40|
| 0.17 | 95.90  | -    |
| 0.18 | 95.62  | 94.40|
| 0.19 | 95.76  | -    |
| 0.20 | 95.62  | 94.40|
| 0.21 | 96.08  | -    |
| 0.22 | 95.84  | 94.40|
| 0.23 | 95.62  | -    |
| 0.24 | 96.04  | 94.40|
| 0.25 | 95.80  | -    |
| 0.26 | 95.88  | 94.00|
| 0.27 | 95.80  | -    |
| 0.28 | 96.02  | 93.80|
| 0.29 | 95.52  | -    |
| 0.30 | 95.44  | -    |
