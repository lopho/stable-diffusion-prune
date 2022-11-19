# (c) 2022 lopho
def prune(
        checkpoint,
        fp16 = False,
        ema = False,
        clip = True,
        vae = True,
):
    sd = checkpoint['state_dict']
    sd_pruned = dict()
    for k in sd:
        cp = k.startswith('model.diffusion_model.')
        cp = cp or (vae and k.startswith('first_stage_model.'))
        cp = cp or (clip and k.startswith('cond_stage_model.'))
        if cp:
            k_in = k
            if ema:
                k_ema = 'model_ema.' + k[6:].replace('.', '')
                if k_ema in sd:
                    k_in = k_ema
            sd_pruned[k] = sd[k_in].half() if fp16 else sd[k_in]
    return { 'state_dict': sd_pruned }

def main(args):
    from argparse import ArgumentParser
    from functools import partial
    parser = ArgumentParser(
            description = "Prune a stable diffusion checkpoint"
    )
    parser.add_argument(
        'input',
        type = str,
        help = "input checkpoint"
    )
    parser.add_argument(
        'output',
        type = str,
        help = "output checkpoint"
    )
    parser.add_argument(
        '-p', '--fp16',
        action = 'store_true',
        help = "convert to float16"
    )
    parser.add_argument(
        '-e', '--ema',
        action = 'store_true',
        help = "use EMA for weights"
    )
    parser.add_argument(
        '-c', '--no-clip',
        action = 'store_true',
        help = "strip CLIP weights"
    )
    parser.add_argument(
        '-a', '--no-vae',
        action = 'store_true',
        help = "strip VAE weights"
    )
    def error(self, message):
        import sys
        sys.stderr.write(f"error: {message}\n")
        self.print_help()
        self.exit()
    parser.error = partial(error, parser) # type: ignore
    args = parser.parse_args(args)
    class pickle:
        import pickle as python_pickle
        class Unpickler(python_pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except:
                    return object
    from torch import save, load
    save(prune(
            load(args.input, pickle_module = pickle), # type: ignore
            fp16 = args.fp16,
            ema = args.ema,
            clip = not args.no_clip,
            vae = not args.no_vae,
    ), args.output)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

