import unittest
from Atlas import Atlas, load_atlas, set_atlas_initial_pose, visualize
from PoseGenerator import PoseGenerator, Trajectory
from pydrake.all import MultibodyPlant, PiecewisePolynomial, PiecewiseQuaternionSlerp, Quaternion
import numpy as np
import time

class TestPoseGenerator(unittest.TestCase):
    def test_simple_trajectory(self):
        plant = MultibodyPlant(0.001)
        load_atlas(plant, add_ground=False)
        T = 2.0
        l_foot_pos_traj = PiecewisePolynomial.FirstOrderHold(np.linspace(0, T, 5), np.array([
            [0.00, 0.09, 0.00],
            [0.25, 0.09, 0.10],
            [0.50, 0.09, 0.00],
            [0.50, 0.09, 0.00],
            [0.50, 0.09, 0.00]]).T)
        r_foot_pos_traj = PiecewisePolynomial.FirstOrderHold(np.linspace(0, T, 5), np.array([
            [0.00, -0.09, 0.00],
            [0.00, -0.09, 0.00],
            [0.00, -0.09, 0.00],
            [0.25, -0.09, 0.10],
            [0.50, -0.09, 0.00]]).T)
        pelvis_pos_traj = PiecewisePolynomial.FirstOrderHold([0.0, T], np.array([
            [0.00, 0.00, Atlas.PELVIS_HEIGHT-0.05],
            [0.50, 0.00, Atlas.PELVIS_HEIGHT-0.05]]).T)
        null_orientation_traj = PiecewiseQuaternionSlerp([0.0, T], [
            Quaternion([1.0, 0.0, 0.0, 0.0]),
            Quaternion([1.0, 0.0, 0.0, 0.0])])

        generator = PoseGenerator(plant, [
            Trajectory('l_foot', Atlas.FOOT_OFFSET, l_foot_pos_traj, 1e-3, null_orientation_traj, 0.05),
            Trajectory('r_foot', Atlas.FOOT_OFFSET, r_foot_pos_traj, 1e-3, null_orientation_traj, 0.05),
            Trajectory('pelvis', np.zeros(3), pelvis_pos_traj, 1e-2, null_orientation_traj, 0.2)])
        for t in np.linspace(0, T, 50):
            q = generator.get_ik(t)
            if q is not None:
                visualize(q)
            else:
                self.fail("Failed to find IK solution!")
            time.sleep(0.2)

if __name__ == "__main__":
    unittest.main()
