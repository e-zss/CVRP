import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
x_lim=np.linspace(-10,10,520)
y_lim=np.linspace(-10,10,520)
z_lim=np.linspace(-10,10,520)
X_points=[]
Y_points=[]
Z_points=[]
for x in x_lim:
    for y in y_lim:
        for z in z_lim:
            if (x**2+(9/4)*y**2+z**2-1)**3-(9/80)*y**2*z**3-x**2*z**3<=0:
                X_points.append(x)
                Y_points.append(y)
                Z_points.append(z)
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X_points,Y_points,Z_points,s=2,alpha=0.5,color='red')
plt.show()
