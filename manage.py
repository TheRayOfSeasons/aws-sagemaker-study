import sys


command = sys.argv[1]


if command == 'test':
    from src.test import test
    test()
elif command == 'deploy':
    from src.endpoint import setup
    setup()
