import pandas as pd
import numpy as np
def dist(a,b):
    dist = np.sqrt(np.sum(np.square(a-b)))
    return dist
a = np.array([2,3,5,6])
b = np.array([1,1,1,1])
d = dist(a,b)
print d

def sort(a):  #  一维数组排序函数
    n = len(a)
    sort = np.array([])
    for i in range(n):
        sort[i] = a[i]		
