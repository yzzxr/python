
from collections.abc import Iterable
from typing import Any


class Singleton(object):
    instance: "Singleton" = None
    initialized: bool = False
    
    def __new__(cls: type):
        if Singleton.instance is None:
            Singleton.instance = super(Singleton, cls).__new__(cls)
        else:
            pass
        return Singleton.instance
    
    def __init__(self) -> None:
        if not self.initialized:
            self.value = 666
            Singleton.initialized = True
            print("created a instance!")
        else:
            pass
        
        
def singleton_decorator(cls: type):
    instance = cls()
    
    def foo(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        else:
            pass
        return instance

    return foo

# @singleton_decorator
class A:
    def __init__(self) -> None:
        self.value = 666
    
    def __add__(self, other: "A") -> "A":
        result: A = A()
        result.value = self.value + other.value
        return result


if __name__ == "__main__":
    
    
