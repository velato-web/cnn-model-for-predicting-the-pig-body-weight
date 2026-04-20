# Pig Weight Prediction using CNN

## Description

This repository contains a CNN-based model for predicting pig body weight using images and optional tabular parameters.

## Related Manuscript

This code is directly related to our manuscript submitted to *The Visual Computer*.

**Please cite this manuscript if you use this code.**

## Requirements

* Python 3.10
* tensorflow
* numpy
* pandas
* scikit-learn
* opencv-python
* matplotlib
* tqdm

Install dependencies:
pip install -r requirements.txt

## Dataset Structure

* Images are stored in subfolders named by animal ID
* CSV file contains:

  * animal_no
  * weight
  * additional parameters

## Usage

### Train model

python train.py

### Predict

python predict.py

## Reproducibility

All experiments can be reproduced using the provided scripts and dataset structure.

## DOI

(To be added after Zenodo upload)

## Citation

If you use this code, please cite our manuscript submitted to *The Visual Computer*.

