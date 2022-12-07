# stable-diffusion-prune
Prune stable diffusion checkpoints.

## Usage
```
usage: prune.py [-h] [-p] [-e] [-c] [-a] [-d] [-u] input output

Prune a stable diffusion checkpoint

positional arguments:
  input           input checkpoint
  output          output checkpoint

optional arguments:
  -h, --help      show this help message and exit
  -p, --fp16      convert to float16
  -e, --ema       use EMA for weights
  -c, --no-clip   strip CLIP weights
  -a, --no-vae    strip VAE weights
  -d, --no-depth  strip depth model weights
  -u, --no-unet   strip UNet weights
```

## Example
Convert to `torch.float16`, use ema weights and remove CLIP model weights.
```sh
python3 prune.py -pec sd-v1-4-full-ema.ckpt pruned.ckpt
```

Keep precision the same and use ema weights.
```sh
python3 prune.py -e sd-v1-4-full-ema.ckpt pruned.ckpt
```

Convert to `torch.float16`, remove VAE and CLIP model weights.
```sh
python3 prune.py -pca sd-v1-4-full-ema.ckpt pruned.ckpt
```

## Dependencies
### Stable diffusion v1
```
numpy
torch!=1.13.0
```
Note that `torch==1.13.0` has a bug in the `torch.load` function that forces you to install `pytorch_lightning` if you want to load stable diffusion checkpoints that include `pytorch_lightning` callbacks. (https://github.com/pytorch/pytorch/issues/88438)

It should be fixed in the next release of torch 1.13.1, 1.14.0-dev or 2.0.0.

### Stable diffusion v2
```
torch
```

