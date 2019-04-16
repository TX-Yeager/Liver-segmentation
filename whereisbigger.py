import numpy as np
import matplotlib.pyplot as plt
#a= np.random.randint(0,10,10)
b=np.array([1, 2,2, 3])
a=np.array([1, 0,2, 1])
#print(max(a(1)))
whichisbetter = np.where(a==b)
print(whichisbetter)
arr = np.array(whichisbetter,dtype=int)
arr.flatten()
print(arr[0])
#print(a)
#print(b)
which = []
for i in range(0, a.size):
    if a[i]>=b[i]:
        which.append(a[i])
    else:
        which.append(b[i])

print(a.size)
print(np.where(a > b, a, b))
plt.figure(figsize=(20, 10))
x = [0,1,2,3]
#plt.plot(x, whichisbetter, 'black', lw=1, marker='.',label="best")
scale_ls = range(0,4)
index_ls = ['CRF', 'Process', 'Only FCN','x']
plt.yticks(scale_ls,index_ls)
plt.plot(x, which, 'black', lw=1, marker='.',label="best")
plt.plot(x, a, 'g', lw=1,marker='.',label="Only FCN")
plt.plot(x, b, 'b', lw=1,marker='.',label="Only FCN")

plt.show()