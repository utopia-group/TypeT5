from typing import List

# A recursive fibonacci function
def fib(n: str) -> List[int]:
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

def t_add(x, y) -> "List[int]":
    r = x + y
    return r

x: int = fib(3)