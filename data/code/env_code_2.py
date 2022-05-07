# Env example 2: some existing annotations

from typing import *


def fib(n: int):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


def foo(bar: int):
    return fib(bar)


class Bar:
    z: str = "hello"
    w: str

    def __init__(self, x: int):
        self.x: int = x
        self.y: Optional[int] = None
        self.reset(self.z)

    def reset(self, w0):
        self.w = w0

    def foo(self, z: str) -> int:
        return self.x + len(z)


bar: Bar = Bar(3)
