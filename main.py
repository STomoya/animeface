
import sys

from importlib import import_module
from utils import argument, debug_mode
from loguru import logger

logger.remove()
logger.add(sys.stdout, level='WARNING')

def main():
    parser = argument.get_default_parser()
    args = parser.parse_known_args()[0]
    if args.debug:
        debug_mode()
        logger.info(args)
    module = import_module(f'.{args.name}', 'implementations')
    module.main(parser)

if __name__ == "__main__":
    main()
