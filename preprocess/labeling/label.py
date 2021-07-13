
import glob
from collections import Counter

import i2v
from PIL import Image
from tqdm import tqdm

'''
Tags
['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair',
'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair', 'long hair',
'short hair', 'twintails', 'drill hair', 'ponytail', 'blush', 'smile', 'open mouth',
'hat', 'ribbon', 'glasses', 'blue eyes', 'red eyes', 'brown eyes', 'green eyes',
'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes', 'orange eyes']
'''

TAGS = ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair',
    'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair', 'long hair',
    'short hair', 'twintails', 'drill hair', 'ponytail', 'blush', 'smile', 'open mouth',
    'hat', 'ribbon', 'glasses', 'blue eyes', 'red eyes', 'brown eyes', 'green eyes',
    'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes', 'orange eyes']

def get_model():
    return i2v.make_i2v_with_chainer(
        '/usr/src/data/illustration2vec/illust2vec_tag_ver200.caffemodel',
        '/usr/src/data/illustration2vec/tag_list.json'
    )

def predict_sort_top(image_path, model, tags=TAGS):
    image = Image.open(image_path)
    tag2prob = model.estimate_specific_tags([image], tags)
    tag2prob = sorted(tag2prob[0].items(), key=lambda x: x[1], reverse=True)
    return tag2prob[0]

def label_it(image_paths, threshold=0.5):

    model = get_model()

    path2tag = {}
    for image in tqdm(image_paths):
        tag2prob = predict_sort_top(image, model)
        if tag2prob[1] > threshold:
            path2tag[image] = tag2prob[0]
    
    return path2tag

def analysis(path2tag):
    tag_list = list(path2tag.values())
    tag_counter = Counter(tag_list)
    print(tag_counter)

def _save(path2tag, filename: str):
    assert filename.endswith('.csv'), 'input filename with ".csv" extension.'

    lines = [','.join([file, tag]) for file, tag in path2tag.items()]
    with open(filename, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(lines))

def animeface():
    threshold = 0.5
    analysis_ = True
    save_filename = '/usr/src/data/animefacedataset/labels.csv'

    image_paths = glob.glob('/usr/src/data/animefacedataset/images/*')
    path2tag = label_it(image_paths, 0.5)
    if analysis_:
        analysis(path2tag)
    _save(path2tag, save_filename)

def danbooru():
    threshold = 0.5

if __name__=='__main__':
    animeface()


