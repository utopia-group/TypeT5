# TypeT5: Seq2seq Type Inference using Static Analysis

This repo contains the source code for the paper [TypeT5: Seq2seq Type Inference using Static Analysis](https://openreview.net/forum?id=4TyNEhI2GdN&noteId=EX_-kP9xah).

## Installation

This project uses [pipenv](https://pipenv.pypa.io/en/latest/) to manage the package dependencies. Pipenv tracks the exact package versions and manages the (project-specific) virtual environment for you. To install all dependencies, make sure you have pipenv installed, then, at the project root, run:
```bash
pipenv sync
```

To add new dependences into the virtual environment, you can manually install them via `pipenv install ..` (using `pipenv`) or `pipenv run pip install ..` (using `pip` from within the virtual environment). If your pytorch installation is not working properly, you might need to install it via `pip` rather than `pipenv`. If you are not using pipenv, make sure to add the environment variables in `.env` to your environment when you run the programs.

You can run all unit tests by running `pipenv run pytest` at the project root.

## Using the trained model
The notebook [scripts/run_typet5.ipynb](scripts/run_typet5.ipynb) shows you how to download the TypeT5 model from Huggingface and then use it to make type predictions for a specified codebase.

## Training a New Model

Dataset
- [scripts/collect_dataset.ipynb](scripts/collect_dataset.ipynb) downloads the creates the BetterTypes4Py dataset used in our paper.

- [scripts/analyze_dataset.ipynb](scripts/analyze_dataset.ipynb) computes basic statistics about our dataset.

- The exact list of repos we used in our paper can be loaded from `data/repos_split.pkl` using `pickle.load`.

Training script
- Run [scripts/train_model.py](scripts/train_model.py) to train a new TypeT5 model. Training takes about 11 hours on a single Quadro RTX 8000 GPU with 48GB memory.


## Development
- Formatter: We use `black` for formatting with the default options.
- Type Checker: We use Pylance for type checking. It's the built-in type checker shipped with the VSCode Python plugin and can be enabled by setting `Python > Anlaysis > Type Checking Mode` to `basic`.
