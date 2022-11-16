
def prune(
        checkpoint,
        clip = False,
        ema = False,
        fp16 = False,
):
    sd = checkpoint['state_dict']
    sd_pruned = dict()
    fp = lambda x: x.half() if fp16 else x
    for k in sd:
        if k.startswith('model.diffusion_model.'):
            if ema:
                k_ema = 'model_ema.' + k[6:].replace('.', '')
                if k_ema in sd:
                    k = k_ema
            sd_pruned[k] = fp(sd[k])
        elif k.startswith('first_stage_model.'):
            sd_pruned[k] = fp(sd[k])
        elif clip and k.startswith('cond_stage_model.'):
            sd_pruned[k] = fp(sd[k])
    return { 'state_dict': sd_pruned }

def main(args):
    from argparse import ArgumentParser
    from functools import partial
    parser: ArgumentParser = ArgumentParser(
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
        help = "convert checkpoint to float16"
    )
    parser.add_argument(
        '-c', '--clip',
        action = 'store_true',
        help = "include the CLIP model in the checkpoint"
    )
    parser.add_argument(
        '-e', '--ema',
        action = 'store_true',
        help = "use ema weights"
    )
    def error(self, message):
        import sys
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        self.exit()
    parser.error = partial(error, parser) # type: ignore
    args = parser.parse_args(args)
    from torch import save, load
    import pickle as python_pickle
    class pickle:
        class Unpickler(python_pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except:
                    return object
    save(prune(
            load(args.input, pickle_module = pickle), # type: ignore
            clip = args.clip,
            ema = args.ema,
            fp16 = args.fp16
    ), args.output)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

