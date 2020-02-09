import argparse


class Hello():
    """
    attributes:
    - name: str
    developer's name
    default=Jack
    """

    def __init__(self, name: str = 'Jack'):
        self.name = 'Jack'
        self.greeting = print(
            'hello, Python! --{}'.format(self.name))

    def __call__(self):
        self.greeting


def parse():
    """Parse Args
    note:
    in ipython, it don't use argparse
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--name', help='your name', default='Jack')
    if hasattr(__builtins__, '__IPYTHON__'):
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    return args


def main():
    args = parse()
    print('[Info] initialize')
    hello = Hello(args.name)
    print('[Info] Call')
    hello()


if __name__ == '__main__':
    main()
