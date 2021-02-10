
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
    return parser
