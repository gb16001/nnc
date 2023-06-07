
from typing import Any


class sensor:
    r'''a sensor,H=1
    '''
    def __init__(self) -> None:
        pass
    def detect(input):
        return input
    def H(input):
        return input
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return args[0]
        