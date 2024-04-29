import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv(r"Airlines.csv")

fig=plt.figure()
#syntax for 3 D projection
ax =plt.axes(projection='3d')


z = np.linspace(0, 1, 200)
x = z * np.sin(30 * z)
y = z * np.cos(30 * z)

# defining the x,y,z axis
ax.set_xlabel('X-axis Length')
ax.set_ylabel('Y-axis Flight')
ax.set_zlabel('Z-axis Time')

#plot 
ax.plot3D(x, y, z, 'purple')
ax.set_title('3D line plotting')
plt.show()