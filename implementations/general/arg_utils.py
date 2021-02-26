
import os
import json
import datetime

def save_args(args, identify=True, id=None):
    args_dict = vars(args)
    if identify:
        if id is None:
            id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        args_file = 'args-{}.json'.format(id)
    else: args_file = 'args.json'
    filename = os.path.join('/usr/src/implementations', args.name, 'result', args_file)
    with open(filename, 'w', encoding='utf-8') as fout:
        json.dump(args_dict, fout, indent=2)
