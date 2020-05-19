
from pathlib import Path

import i2v
from PIL import Image
from tqdm import tqdm

base = Path('/usr/src/data/images')
image_paths = list(base.glob('*'))

illust2vec = i2v.make_i2v_with_chainer('/usr/src/data/illustration2vec/illust2vec_tag_ver200.caffemodel', '/usr/src/data/illustration2vec/tag_list.json')

'''
Tags
Some of the tags are eliminated, because of having too small number of samples. (Threshold : #samples > 50)

Original
['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair',
'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair', 'long hair',
'short hair', 'twintails', 'drill hair', 'ponytail', 'blush', 'smile', 'open mouth',
'hat', 'ribbon', 'glasses', 'blue eyes', 'red eyes', 'brown eyes', 'green eyes',
'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes', 'orange eyes']

->

Changed
['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair',
'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair', 'long hair',
'short hair',                            'ponytail', 'blush', 'smile', 'open mouth',
'hat',           'glasses', 'blue eyes', 'red eyes', 'brown eyes', 'green eyes',
'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes'               ]
'''

tags = ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair',
        'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair', 'long hair',
        'short hair', 'ponytail', 'blush', 'smile', 'open mouth',
        'hat', 'glasses', 'blue eyes', 'red eyes', 'brown eyes', 'green eyes',
        'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes']

lines = []

for path in tqdm(image_paths):
    # extract year from path (**/<id>_<year>.jpg)
    year = path.name.split('.')[0].split('_')[-1]

    image = Image.open(path)
    # extract
    tag2prob = illust2vec.estimate_specific_tags([image], tags)

    # skip when no response
    if tag2prob == []:
        print(path, 'has been skipped')
        continue

    # sort dict by value
    sorted_tag2prob = sorted(tag2prob[0].items(), key=lambda x: x[1], reverse=True)

    line = ','.join([str(path), sorted_tag2prob[0][0].replace(' ', '_'), year])    
    lines.append(line)

with open('/usr/src/data/labels.csv', 'w', encoding='utf-8') as fout:
    fout.write('\n'.join(lines))

# analysis
from collections import Counter
from pprint import pprint
print('Images : {} / {}'.format(len(lines), len(image_paths)))
i2vtag_label_list = [line.split(',')[1] for line in lines]
year_label_list = [line.split(',')[2] for line in lines]

i2vtag_counter = Counter(i2vtag_label_list)
year_counter = Counter(year_label_list)

pprint(i2vtag_counter)
pprint(year_counter)

