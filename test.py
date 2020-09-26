import random 
dict = {}

for i in range(10):
    dict[(i, i)] = 1

print(list(dict.keys())[1])
for i in range(10, 1, -1):
    print(i)