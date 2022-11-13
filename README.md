# TypeT5: Seq2seq Type Inference using Static Analysis

This repo contains the source code for the paper [TypeT5: Seq2seq Type Inference using Static Analysis](TODO).

## Installation

## Running Trained Model
- TODO to download the pre-trained model weights.

## Dataset

- [scripts/collect_dataset.ipynb](scripts/collect_dataset.ipynb) downloads the creates the BetterTypes4Py dataset used in our paper.

- [scripts/analyze_dataset.ipynb](scripts/analyze_dataset.ipynb) computes basic statistics about our dataset.

## Training a New Model

- Run [scripts/train_model.py](scripts/train_model.py) to train a new TypeT5 model. Training takes about 11 hours on a single Quadro RTX 8000 GPU with 48GB memory.
