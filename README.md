# stable-diffusion-prune
Prune stable diffusion checkpoints.

## Usage
```
usage: prune.py [-h] [-p] [-c] [-e] input output

Prune a stable diffusion checkpoint

positional arguments:
  input       input checkpoint
  output      output checkpoint

optional arguments:
  -h, --help  show this help message and exit
  -p, --fp16  convert checkpoint to float16
  -c, --clip  include the CLIP model in the checkpoint
  -e, --ema   use ema weights
```

## Example
Convert to `torch.float16`, use ema weights and remove CLIP model weights.
```sh
python3 prune.py -pe sd-v1-4-full-ema.ckpt pruned.ckpt
```

Keep precision the same, use ema weights and include CLIP model weights.
```sh
python3 prune.py -ce sd-v1-4-full-ema.ckpt pruned.ckpt
```

## Dependencies
```
numpy
torch!=1.13.0
```
Note that `torch==1.13.0` has a bug in the `torch.load` function that forces you to install `pytorch_lightning` if you want to load stable diffusion checkpoints that include `pytorch_lightning` callbacks. (https://github.com/pytorch/pytorch/issues/88438)

It should be fixed in the next release of torch 1.13.1 or 1.14.0.
