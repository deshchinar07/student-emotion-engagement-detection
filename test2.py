import matplotlib.pyplot as plt
import numpy as np

file = open(r"/Users/deshc/Desktop/Chinar/Research/Engagement Level Detection/School Videos/Lecture 2.txt", "r").read()
a = file.split("\n")
arr1 = []
arr2 = []
fCnt = 0
cnt = 0

res = [0, 0, 0, 0, 0]
res2 = [0, 0, 0, 0, 0]
for i in range(len(a)):
    print(cnt)
    engagementLevel = int(a[i][-1])
    arr1.append(cnt/12)
    cnt = cnt + 1
    arr2.append(engagementLevel)
    res2[engagementLevel - 1] = res2[engagementLevel - 1] + 1
    res[engagementLevel - 1] = res[engagementLevel - 1] + 1/12
    
print(res)
print(res2[0]/len(arr1))
print(res2[1]/len(arr1))
print(res2[2]/len(arr1))
print(res2[3]/len(arr1))
print(res2[4]/len(arr1))

print(arr2)

plt.plot(np.array(arr1), np.array(arr2))
plt.ylim(1, 5)
plt.yticks([1, 2, 3, 4, 5])
plt.xlabel("Time (min)")
plt.ylabel("Engagement Level")
plt.show()
