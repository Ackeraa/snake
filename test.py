dict = {}

for i in range(10):
    dict[i] = 1

del dict[1]
print(list(dict.keys())[1])

a = [1, 2]
b = [3, 4]
c = (1, 2)
d = list(c)
print(d)