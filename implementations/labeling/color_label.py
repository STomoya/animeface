
import random
import glob
from collections import Counter

import i2v
from PIL import Image
from tqdm import tqdm
import numpy as np

'''
Tags:
hair color
['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair',
 'green hair', 'red hair', 'silver hair']
eye color
['blue eyes', 'red eyes', 'brown eyes', 'green eyes', 'purple eyes',
 'yellow eyes', 'pink eyes']
'''

HAIR_TAGS = ['blonde hair', 'brown hair', 'black hair', 'blue hair',
    'pink hair', 'purple hair', 'green hair', 'red hair', 'silver hair']
EYE_TAGS  = ['blue eyes', 'red eyes', 'brown eyes', 'green eyes', 'purple eyes',
    'yellow eyes', 'pink eyes']
GLASS_TAG = ['glasses']

def get_model():
    return i2v.make_i2v_with_chainer(
        '/usr/src/data/illustration2vec/illust2vec_tag_ver200.caffemodel',
        '/usr/src/data/illustration2vec/tag_list.json'
    )


def label_it(
    image_paths, tags, model, num_images,
    threshold=0.5, used_images: list=[]
):
    '''image labeling for category with multiple tags'''
    path2tag = []
    bar = tqdm(total=num_images)
    for image in tqdm(image_paths):
        if not image in used_images:
            # predict
            tag2prob = model.estimate_specific_tags([Image.open(image)], tags)[0]
            # sort by probability and get top
            # tag2prob 0: tag name, 1: probability
            tag2prob = sorted(tag2prob.items(), key=lambda x: x[1], reverse=True)[0]
            if tag2prob[1] > threshold:
                path2tag.append((image, tag2prob[0]))
                bar.update(1)
        # end
        if len(path2tag) == num_images:
            break
    return path2tag

def label_binary(
    image_paths, tag, model, num_images,
    w_threshold=0.5, wo_threshold=0.01, used_images: list=[], balanced=True
):
    '''image labeling for binary category'''
    assert len(tag) == 1, ''
    path2tag_with = []
    path2tag_without = []
    per_tag = num_images // 2
    bar = tqdm(total=num_images)
    for image in tqdm(image_paths):
        # skip used images
        if not image in used_images:
            # predict
            tag2prob = model.estimate_specific_tags([Image.open(image)], tag)[0]
            prob = tag2prob[tag[0]]
            # over threshold, less than per_tag
            if prob > w_threshold \
                and len(path2tag_with) < per_tag:
                path2tag_with.append((image, 'with'))
                bar.update(1)
            # under threshold, less than per_tag
            elif prob < wo_threshold \
                and len(path2tag_without) < per_tag:
                path2tag_without.append((image, 'without'))
                bar.update(1)
        # end
        if len(path2tag_with) + len(path2tag_without) == num_images:
            break

    if balanced:
        # make tags have same samples
        less = min(len(path2tag_with), len(path2tag_without))
        path2tag_with    = path2tag_with[:less]
        path2tag_without = path2tag_without[:less]

    return path2tag_with + path2tag_without
    
def label_them(image_paths, per_category):
    '''label images to all categories with no overlap'''
    model = get_model()
    # label glasses
    random.shuffle(image_paths)
    path2tag_glass = label_binary(image_paths, GLASS_TAG, model, per_category)
    # label hair color
    random.shuffle(image_paths)
    used_images = [path for (path, _) in path2tag_glass]
    path2tag_hair = label_it(image_paths, HAIR_TAGS, model, per_category, used_images=used_images)
    # label eye color
    random.shuffle(image_paths)
    used_images.extend([path for (path, _) in path2tag_hair])
    path2tag_eye = label_it(image_paths[::-1], EYE_TAGS, model, per_category, used_images=used_images)
    
    return path2tag_hair, path2tag_eye, path2tag_glass

def _save(path2tag, filename):
    '''save labels to csv file'''
    assert filename.endswith('.csv')
    lines = [','.join(list(labeled)) for labeled in path2tag]
    with open(filename, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(lines))

def animeface():
    hair_filename = '/usr/src/data/animefacedataset/hair_color_labels.csv'
    eye_filename = '/usr/src/data/animefacedataset/eye_color_labels.csv'
    glass_filename = '/usr/src/data/animefacedataset/glass_labels.csv'
    per_category = 10000

    image_paths = glob.glob('/usr/src/data/animefacedataset/images/*')
    assert len(image_paths) > per_category*3
    hair, eye, glass = label_them(image_paths, per_category)
    _save(hair, hair_filename)
    _save(eye, eye_filename)
    _save(glass, glass_filename)

def danbooru():
    hair_filename = '/usr/src/data/danbooru/portraits/hair_color_labels.csv'
    eye_filename = '/usr/src/data/danbooru/portraits/eye_color_labels.csv'
    glass_filename = '/usr/src/data/danbooru/portraits/glass_labels.csv'
    per_category = 50000

    image_paths = glob.glob('/usr/src/data/danbooru/portraits/portraits/*')
    assert len(image_paths) > per_category*3
    hair, eye, glass = label_them(image_paths, per_category)
    _save(hair, hair_filename)
    _save(eye, eye_filename)
    _save(glass, glass_filename)

if __name__=='__main__':
    # animeface()
    danbooru()