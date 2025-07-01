import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

Np=200
Nn=200
X=np.r_[np.random.multivariate_normal([-5,0],[[1,0],[0,1]],Np),
        np.random.multivariate_normal([5,0],[[1,0],[0,1]],Nn)]
y=np.concatenate([np.repeat(-1,Np),np.repeat(1,Nn)])

Xte=np.r_[np.random.multivariate_normal([-5,0],[[1,0],[0,1]],Np),
        np.random.multivariate_normal([5,0],[[1,0],[0,1]],Nn)]
yte=np.concatenate([np.repeat(-1,Np),np.repeat(1,Nn)])

max_iter=1000
tol=1e-6
lmd=1.
step_size=1e-4
Xtr=np.c_[np.ones(len(y)),X]
w=np.zeros(Xtr.shape[1])
for iteration in xrange(max_iter):
    wold=w
    i=np.random.randint(0,len(y))
    xi,yi=Xtr[i],y[i]
    if (1-yi*w.dot(xi))>=0:
        tmp=-yi*xi
    else:
        tmp=0.
    w=w-step_size*(2*lmd*w+tmp)
    if np.sum((w-wold)**2)<=tol:
        break

Xte=np.c_[np.ones(len(yte)),Xte]
label=2*(Xte.dot(w)>=0)-1
print "accuracy:",np.sum(label==yte)/np.float(len(yte))

colors=np.array(["r","b"])
cy=1*(y==1)
plt.scatter(X[:,0],X[:,1],color=colors[cy])

nx,ny=200,200
x_min, x_max=plt.xlim()
y_min, y_max=plt.ylim()
xx, yy=np.meshgrid(np.linspace(x_min, x_max, nx),
                   np.linspace(y_min, y_max, ny))
PX=np.c_[xx.ravel(), yy.ravel()]
PX=np.c_[np.ones(PX.shape[0]),PX]
Z=PX.dot(w)
Z=Z.reshape(xx.shape)
plt.contour(xx, yy, Z, [0], linewidths=2., colors='green')
plt.tight_layout()
plt.show()