import numpy as np
queue = list()

test = [1,2,3]
test1 = [4,5,6]

queue.append(test)
queue.append(test1)


v1 = queue.pop()
v2 = queue.pop()

v3 = np.array([v1]) -np.array([v2])
queue.append(v3)
test = queue.pop()
print(type(test))
v3 = v3 / np.linalg.norm(v3, axis=1)[:, np.newaxis]


print(v3[0])