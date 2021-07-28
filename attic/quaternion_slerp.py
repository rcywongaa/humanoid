from pydrake.all import PiecewiseQuaternionSlerp, Quaternion
import numpy as np

q1 = np.array([1.0, 0.0, 0.0, 0.0])
q2 = np.array([-0.5, -0.5, 0.5, 0.5])
slerp1 = PiecewiseQuaternionSlerp(breaks=[0, 1], quaternions=[Quaternion(q1), Quaternion(q2)])
for i in np.linspace(0, 1, 11):
    print(f"1: {Quaternion(slerp1.value(i)).wxyz()}")

q3 = np.array([1.0, 0.0, 0.0, 0.0])
q4 = np.array([-0.5, -0.5, 0.4999999999999999, 0.5000000000000001])
slerp2 = PiecewiseQuaternionSlerp(breaks=[0, 1], quaternions=[Quaternion(q3[0:4]), Quaternion(q4[0:4])])
for i in np.linspace(0, 1, 11):
    print(f"2: {Quaternion(slerp2.value(i)).wxyz()}")
