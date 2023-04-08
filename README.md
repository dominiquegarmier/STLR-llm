[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/DominiqueGarmier/STLR-llm/main.svg)](https://results.pre-commit.ci/latest/github/DominiqueGarmier/STLR-llm/main) [![code style black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Work in progress...

# STLR-llm

short-ontext transformer long-context recurrent large language model. Use Tranformers to pick up local bidirectional relationships and RNNs to capture long-range dependencies. Potentally offering the best of both worlds. Including unlimited context.

## Aknowledgements

This project was inspired by the remarkable power of RNNs showcased by [RWKV](https://github.com/BlinkDL/RWKV-LM).

## Installation

```
git clone git@github.com:DominiqueGarmier/STLR-llm.git
cd STLR-llm

pip install -r requirements.txt
python -m stlr
```

and if you would like to contribute:

```
pip install -r requirements-dev.txt
pre-commit install
```

## About

This is just me putting down some thoughs. This Model has not been trained yet nor evaluted. I just had some ideas about an architecture that I wanted ot implement. I still have some more ideas I would like to implement before I worry about training anything.

Some other things to note:

- Currently the Hyperparameters in `STLRConfig` are more or less chosen at random. So parameter counts are meaningless.
- The model does not yet feature a tokenizer or tokenembedder (use one from huggingface) aswell as a positional encoder.
- There are some considerations to be made about efficiently parallelizing the model across batches (mostly for varying sequence lengths across batches)

More than anything this was an exercise in implementing a model from scratch.
