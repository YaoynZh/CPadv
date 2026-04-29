# README

## Overview
This project, titled "CPadv: ***[CPadv: Black-Box Adversarial Attack for Time Series Classification Based on Change Point Detection]***," includes a demo dataset called "ECG200" along with pre-trained model checkpoints for FCN and ResNet. For additional datasets, you can download any of the UCR datasets from [this link](http://www.timeseriesclassification.com/).

## Data Folder Structure
- The _TEST.txt file has been stratified into two subsets: _cp.txt and _attack.txt.
- _TRAIN.txt is used to train the target model.
- _cp.txt contains data for generating GMM intervals.
- _attack.txt is the dataset intended for attack simulations.

## Installation
To install the necessary dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Quick Start with CPadv
To quickly run the CPadv attack:
1. Use the 'main.py' file, which initiates the attack process on the 'ECG200_attack.txt' dataset.
eg.
```bash
python main.py
```
2. The intervals used for the attack are located in the `cp_pos` folder.
3. You can customize the 'config' parameters in the 'main.py' file as needed.
