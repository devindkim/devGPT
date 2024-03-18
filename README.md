# devGPT

A simple pytorch implementation of GPT-2, optimized to run on Macbook Pro M1/M2. You can get "reasonable" generations on most text datasets from one overnight run.

## quick start

```
$ pip install -r requirements.txt
```

By default, trains character-level GPT on the script of Avatar: The Last Airbender.

```
$ python data/avatar/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory.

```
$ python train.py
$ python sample.py
```
