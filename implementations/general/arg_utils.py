
import os
import json

def save_args(args):
    args_dict = vars(args)
    filename = os.path.join('/usr/src/implementations', args.name, 'result', 'args.json')
    with open(filename, 'w', encoding='utf-8') as fout:
        json.dump(args_dict, fout, indent=2)
