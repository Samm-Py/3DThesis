import numpy as np
from math import sin, cos

x, y = -1, -1

angle = np.pi-np.arctan2(x,y)

print(angle*180/3.1415)

# Apply the rotation to all points
xRot, yRot = x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle)

print(xRot,yRot)
