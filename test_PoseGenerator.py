import unittest
from Atlas import Atlas, load_atlas, set_atlas_initial_pose, visualize
from PoseGenerator import PoseGenerator
from pydrake.all import MultibodyPlant, PiecewisePolynomial, PiecewiseQuaternionSlerp, Quaternion
import numpy as np
import time

class TestPoseGenerator(unittest.TestCase):
    def test_simple_trajectory(self):
        plant = MultibodyPlant(0.001)
        load_atlas(plant, add_ground=False)
        l_foot_pos_traj = PiecewisePolynomial.FirstOrderHold([0.0, 1.0, 2.0, 3.0, 4.0], np.array([
            [0.00, 0.09, 0.00],
            [0.25, 0.09, 0.10],
            [0.50, 0.09, 0.00],
            [0.50, 0.09, 0.00],
            [0.50, 0.09, 0.00]]).T)
        r_foot_pos_traj = PiecewisePolynomial.FirstOrderHold([0.0, 1.0, 2.0, 3.0, 4.0], np.array([
            [0.00, -0.09, 0.00],
            [0.00, -0.09, 0.00],
            [0.00, -0.09, 0.00],
            [0.25, -0.09, 0.10],
            [0.50, -0.09, 0.00]]).T)
        pelvis_pos_traj = PiecewisePolynomial.FirstOrderHold([0.0, 4.0], np.array([
            [0.00, 0.00, Atlas.PELVIS_HEIGHT-0.1],
            [0.50, 0.00, Atlas.PELVIS_HEIGHT-0.1]]).T)
        pelvis_orientation_traj = PiecewiseQuaternionSlerp([0.0, 4.0], [
            Quaternion([1.0, 0.0, 0.0, 0.0]),
            Quaternion([1.0, 0.0, 0.0, 0.0])])

        generator = PoseGenerator(plant, {
            'l_foot': [l_foot_pos_traj, None],
            'r_foot': [r_foot_pos_traj, None],
            'pelvis': [pelvis_pos_traj, pelvis_orientation_traj]})
        q_guess = None
        for t in np.linspace(0, 4.0, 50):
            q = generator.get_ik(t, q_guess)
            if q is not None:
                q_guess = q
                visualize(q)
            time.sleep(0.2)

if __name__ == "__main__":
    unittest.main()
