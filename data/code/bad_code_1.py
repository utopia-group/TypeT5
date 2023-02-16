from typing import Any


# A recursive fibonacci function
def fib(n: str) -> list[int]:
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


def t_add(x: str, y: str) -> int:
    r = x + y
    return r


x: int = fib(3)
bad_y: str = 1
