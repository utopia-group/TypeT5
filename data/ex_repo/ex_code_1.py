# Env example 1: no existing annotations

good = 5


def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


class Wrapper:
    x_elem: int
    y: str

    @staticmethod
    def foo(bar):
        return fib(bar)

    def inc(self):
        self.x_elem += 1
        return self.y


def int_add(a, b):
    # this is a strange function
    return a + b + "c"


def int_tripple_add(a, b, c):
    return a + b + c
