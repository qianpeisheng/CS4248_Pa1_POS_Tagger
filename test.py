import numpy as np 
x = ['1','aa']
xx = np.array(x)
qq = np.zeros([2,3], dtype = float)
tt = xx.reshape((2,1))
oo = np.hstack([tt,qq])
vv = np.array([0,4,5,6])
ttt = np.vstack([vv,oo])


#print(ttt['0'])

print([1,1,2].index(1))
x = set([1,2])
x = x.union({3})
print(x)