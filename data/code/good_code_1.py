import math
from typing import Any  # [added by SPOT]
from typing import Optional

print(math.sin(4))

x_str: str = "x"
y: Any = 1
z_str: str = x_str + y


class Foo:
    def __init__(self, x: int):
        self.x: int = x
        self.y: Optional[int] = None
        self.z = "hello"

    def foo(self, z: str) -> int:
        return self.x + len(z)
