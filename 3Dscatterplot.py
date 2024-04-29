import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv(r"Airlines.csv")

# Assuming the DataFrame contains columns named 'X', 'Y', and 'Z' for coordinates
x = df['Length']
y = df['Flight']
z = df['Time']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(x, y, z, c='purple', marker='o')

# Labels
ax.set_xlabel('X-axis Length')
ax.set_ylabel('Y-axis Flight')
ax.set_zlabel('Z-axis Time')

# Title
ax.set_title('3D Scatter Plot')

plt.show()
