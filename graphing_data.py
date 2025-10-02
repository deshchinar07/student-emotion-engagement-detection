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
    line = a[i].split(" ")
    if len(line) > 2:
        frameNo = int(line[1])
        engagementLevel = int(line[8])
        if frameNo != fCnt:
            arr1.append(cnt*1.6/60)
            cnt = cnt + 1
            arr2.append(engagementLevel)
            res2[engagementLevel - 1] = res2[engagementLevel - 1] + 1
            res[engagementLevel - 1] = res[engagementLevel - 1] + 1.6/60
            fCnt = frameNo


plt.plot(np.array(arr1), np.array(arr2))
plt.ylim(1, 5)
plt.yticks([1, 2, 3, 4, 5])
plt.xlabel("Time (min)")
plt.ylabel("Engagement Level")
plt.show()
