# Improving Neural Story Generation by Targeted Common Sense Grounding
This repository contains the code to replicate the paper "Improving Neural Story Generation by Targeted Common Sense Grounding".

## Environment Setup
We use Docker to ensure a consistent development environment.
First, ensure Docker and NVIDIA-Docker is installed.

Build Docker image:
```
docker build -t storygen .
```

Run bash shell in image:
```
docker run --rm -w /src -v $(pwd):/src storygen /bin/bash
```
Now you can run scripts within the shell.

For all scripts you will need to download the corresponding datasets before running.

## Training
To train a model, run the following. See `--help` for CLI argument options.
```
python train.py [experiment_name]
```

## Evaluation
Generate text from model
```
python -m analysis.generate.py
```

Compute perplexity from model
```
python -m analysis.eval_ppl.py
```

Compute prompt ranking accuracy from model
```
python -m analysis.eval_prompt_rank.py
```

Compute common sense reasoning accuracy from model
```
python -m analysis.eval_csr.py
```

## Attribution
If you use this code in your research, cite our paper via the following BibTeX.

```
@inproceedings{mao2019emnlp,
  title={Improving Neural Story Generation by Targeted Common Sense Grounding},
  author={Mao, Huanru Henry and Majumder, Bodhisattwa Prasad and McAuley, Julian and Cottrell, Garrison W.},
  booktitle={EMNLP},
  year={2019}
}
```