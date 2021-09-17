
from loguru import logger

from rich.console import Console
from rich.logging import RichHandler
from rich import traceback
from rich import print as richprint, inspect

__all__ = [
    'print',
    'console',
    'inspect'
]

# set print to rich.print
builtin_print = print
print = richprint

# rich traceback
console = Console()
traceback.install(console=console)

# rich logging with loguru
logger.remove()
logger.add(RichHandler(console=console), format='{message}')
