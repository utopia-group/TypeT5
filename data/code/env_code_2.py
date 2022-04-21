# Env example 2: some existing annotations

from typing import Optional

def fib(n: int):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

def foo(bar: int):
    return fib(bar)


class Bar:
    z: str = "hello"
    def __init__(self, x: int):
        self.x: int = x
        self.y: Optional[int] = None

    def foo(self, z: str) -> int:
        return self.x + len(z)