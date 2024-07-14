from dis import dis


def f():
    data = []
    data1 = []
    def foo(value: object):
        data.append(value)
        data1.append(value)
        return data
    return foo

g = f()

print(g.__closure__)
print(g(1))
print(g(2))



print("hello")
print("hello")
print("hello")
print("hello")
print("hello")