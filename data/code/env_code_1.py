# Env example 1: no existing annotations

def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

def foo(bar):
    return fib(bar)

def int_add(a, b):
    return a + b + "c"

def int_tripple_add(a, b, c):
    return a + b + c