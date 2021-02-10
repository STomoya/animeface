
from importlib import import_module
from parser import get_common_parser

def main():
    parser = get_common_parser()
    args, _ = parser.parse_known_args()
    module = import_module(f'.{args.name}', 'implementations')
    module.main(parser)


if __name__ == "__main__":
    main()