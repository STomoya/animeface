
import argparse

def get_common_parser():
    '''parser with common arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Name of implementation')
    parser.add_argument('--disable-gpu', action='store_true', default=False, help='Disable GPU')
    parser.add_argument('--disable-amp', action='store_true', default=False, help='Disable AMP')
    parser.add_argument('--dataset', default='animeface', choices=['animeface', 'danbooru'], help='Dataset name')
    parser.add_argument('--image-size', default=128, type=int, help='Size of image')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
    parser.add_argument('--min-year', default=2005, type=int, help='Minimum of generated year. Ignored when dataset==danbooru')
    parser.add_argument('--num-images', default=60000, type=int, help='Number of images to include in training set. Ignored when dataset==animeface')
    parser.add_argument('--save', default=1000, type=int, help='Interval for saving the model. Also used for sampling images and saving them for qualitative eval.')
    parser.add_argument('--max-iters', default=-1, type=int, help='The maximum iteration to train the model. If < 0, it will be calculated using "--default-epochs"')
    parser.add_argument('--default-epochs', default=100, type=int, help='Used to calculate the max iteration if "--max-iters" < 0')
    return parser
