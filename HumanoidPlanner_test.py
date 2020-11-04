from HumanoidPlanner import HumanoidPlanner
from HumanoidPlanner import create_q_interpolation, create_r_interpolation, apply_angular_velocity_to_quaternion
import numpy as np
from Atlas import Atlas, load_atlas, set_atlas_initial_pose
import unittest
import pdb

from pydrake.all import MultibodyPlant
from pydrake.autodiffutils import initializeAutoDiff

mbp_time_step = 1.0e-3
epsilon = 1e-5

def assert_autodiff_array_almost_equal(autodiff_array, float_array):
    float_array = np.array([i.value() for i in autodiff_array])
    np.testing.assert_array_almost_equal(float_array, float_array)

def default_q():
    pelvis_orientation = [1., 0., 0., 0.]
    pelvis_position = [0., 0., 0.93845] # Feet just touching ground
    joint_positions = [0.] * Atlas.NUM_ACTUATED_DOF
    return np.array(pelvis_orientation + pelvis_position + joint_positions)

def default_v():
    pelvis_rotational_velocity = [0., 0., 0.]
    pelvis_linear_velocity = [0., 0., 0.]
    joint_velocity = [0.] * Atlas.NUM_ACTUATED_DOF
    return np.array(pelvis_rotational_velocity + pelvis_rotational_velocity + joint_velocity)

class TestHumanoidPlannerStandalone(unittest.TestCase):
    def test_create_q_interpolation(self):
        plant = MultibodyPlant(mbp_time_step)
        load_atlas(plant, add_ground=False)
        context = plant.CreateDefaultContext()
        pass

    def test_create_r_interpolation(self):
        pass

    def test_apply_angular_velocity_to_quaternion_float(self):
        q = np.array([1., 0., 0., 0.])
        t = 1.0
        w = np.array([1., 0., 0.])
        q_new = apply_angular_velocity_to_quaternion(q, w, t)
        q_new_expected = np.array([0.877583, 0.479425, 0.0, 0.0])
        np.testing.assert_array_almost_equal(q_new, q_new_expected)

        w = np.array([0., 2., 0.])
        t = 0.5
        q_new = apply_angular_velocity_to_quaternion(q, w, t)
        q_new_expected = np.array([0.877583, 0.0, 0.479425, 0.0])
        np.testing.assert_array_almost_equal(q_new, q_new_expected)

        w = np.array([0., 0., 0.5])
        t = 2.0
        q_new = apply_angular_velocity_to_quaternion(q, w, t)
        q_new_expected = np.array([0.877583, 0.0, 0.0, 0.479425])
        np.testing.assert_array_almost_equal(q_new, q_new_expected)

        w = np.array([1., 2., 3.])
        t = 1.0
        q_new = apply_angular_velocity_to_quaternion(q, w, t)
        q_new_expected = np.array([0.13245323570650439, -0.26490647141300877, -0.5298129428260175, -0.7947194142390264])
        np.testing.assert_array_almost_equal(q_new, q_new_expected)

    def test_apply_angular_velocity_to_quaternion_AutoDiffXd(self):
        # Because initializeAutoDiff automatically converts to 2D arrays
        q_ad = initializeAutoDiff(np.array([1., 0., 0., 0.])).flatten()
        w_ad = initializeAutoDiff(np.array([1., 0., 0.])).flatten()
        t_ad = initializeAutoDiff([1.0]).flatten()[0]
        q_new_ad = apply_angular_velocity_to_quaternion(q_ad, w_ad, t_ad)
        q_new_expected = np.array([0.877583, 0.479425, 0.0, 0.0])
        assert_autodiff_array_almost_equal(q_new_ad, q_new_expected)

class TestHumanoidPlanner(unittest.TestCase):
    def setUp(self):
        self.plant = MultibodyPlant(mbp_time_step)
        load_atlas(self.plant, add_ground=False)
        self.context = self.plant.CreateDefaultContext()
        upright_context = self.plant.CreateDefaultContext()
        set_atlas_initial_pose(self.plant, upright_context)
        q_nom = self.plant.GetPositions(upright_context)
        self.planner = HumanoidPlanner(self.plant, Atlas.CONTACTS_PER_FRAME, q_nom)

    def test_getPlantAndContext_float(self):
        pelvis_orientation = [1., 0., 0., 0.]
        pelvis_position = [5., 6., 7.]
        joint_positions = [20 + i for i in range(Atlas.NUM_ACTUATED_DOF)]
        q = np.array(pelvis_orientation + pelvis_position + joint_positions)
        pelvis_rotational_velocity = [8., 9., 10.]
        pelvis_linear_velocity = [11., 12., 13.]
        joint_velocity = [100 + i for i in range(Atlas.NUM_ACTUATED_DOF)]
        v = np.array(pelvis_rotational_velocity + pelvis_rotational_velocity + joint_velocity)
        plant, context = self.planner.getPlantAndContext(q, v)
        np.testing.assert_array_almost_equal(plant.GetPositions(context), q)
        np.testing.assert_array_almost_equal(plant.GetVelocities(context), v)

    def test_getPlantAndContext_AutoDiffXd(self):
        pelvis_orientation = [1., 0., 0., 0.]
        pelvis_position = [5., 6., 7.]
        joint_positions = [20 + i for i in range(Atlas.NUM_ACTUATED_DOF)]
        q = np.array(pelvis_orientation + pelvis_position + joint_positions)
        q_ad = initializeAutoDiff(q).flatten()
        pelvis_rotational_velocity = [8., 9., 10.]
        pelvis_linear_velocity = [11., 12., 13.]
        joint_velocity = [100 + i for i in range(Atlas.NUM_ACTUATED_DOF)]
        v = np.array(pelvis_rotational_velocity + pelvis_rotational_velocity + joint_velocity)
        v_ad = initializeAutoDiff(v).flatten()
        plant_ad, context_ad = self.planner.getPlantAndContext(q_ad, v_ad)
        q_result_ad = plant_ad.GetPositions(context_ad)
        v_result_ad = plant_ad.GetVelocities(context_ad)
        assert_autodiff_array_almost_equal(q_result_ad, q)
        assert_autodiff_array_almost_equal(v_result_ad, v)

    def test_toTauj(self):
        tau_k = [i for i in range(16)]
        tau_j = self.planner.toTauj(tau_k)
        tau_j_expected = np.array([[0.0, 0.0, i] for i in range(16)])
        np.testing.assert_array_almost_equal(tau_j, tau_j_expected)

    def test_get_contact_position(self):
        pass

    def test_get_contact_positions_z(self):
        pelvis_orientation = [1., 0., 0., 0.]
        pelvis_position = [0., 0., 0.93845]
        joint_positions = [0.] * Atlas.NUM_ACTUATED_DOF
        q = np.array(pelvis_orientation + pelvis_position + joint_positions)
        pelvis_rotational_velocity = [0., 0., 0.]
        pelvis_linear_velocity = [0., 0., 0.]
        joint_velocity = [0.] * Atlas.NUM_ACTUATED_DOF
        v = np.array(pelvis_rotational_velocity + pelvis_rotational_velocity + joint_velocity)
        contact_positions_z = self.planner.get_contact_positions_z(q, v)
        expected_contact_positions_z = [0.] * Atlas.NUM_CONTACTS
        np.testing.assert_allclose(contact_positions_z, expected_contact_positions_z, atol=epsilon)

    def test_calc_h(self):
        pass

    def test_calc_r(self):
        pass

    def test_eq7c(self):
        q = default_q()
        v = default_v()
        h = np.array([0., 0., 0.])
        q_v_h = np.concatenate([q, v, h])
        np.testing.assert_allclose(self.planner.eq7c(q_v_h), 0.)

    def test_eq7d(self):
        q = default_q()
        qprev = default_q()
        v = default_v()
        dt = [0.5]
        q_qprev_v_dt = np.concatenate([q, qprev, v, dt])
        np.testing.assert_allclose(self.planner.eq7d(q_qprev_v_dt), 0.)

        pelvis_rotational_velocity = [1., 2., 3.]
        pelvis_linear_velocity = [1., 2., 3.]
        joint_velocity = [i for i in range(Atlas.NUM_ACTUATED_DOF)]
        v = np.array(pelvis_rotational_velocity + pelvis_linear_velocity + joint_velocity)
        # pelvis_orientation = [0.7030059015585651, -0.15777337848854225, 0.487880743779453, 0.4928109609813361]
        # pelvis_orientation = [0.541, 0.475, 0.192, 0.666]
        pelvis_orientation = [0.0, 0.0, 0.0, 0.0]
        pelvis_position = [0.5, 1., 1.5+0.93845]
        joint_positions = [i*0.5 for i in range(Atlas.NUM_ACTUATED_DOF)]
        q = np.array(pelvis_orientation + pelvis_position + joint_positions)
        q_qprev_v_dt = np.concatenate([q, qprev, v, dt])
        np.testing.assert_allclose(self.planner.eq7d(q_qprev_v_dt), 0., atol=epsilon)

    def test_eq7h(self):
        pass

    def test_eq7i(self):
        pass

    def test_eq8a_lhs(self):
        pass

    def test_eq8b_lhs(self):
        pass

    def test_eq8c_2(self):
        pass

    def test_eq9a_lhs(self):
        pass

    def test_eq9b_lhs(self):
        pass

    def test_pose_error_cost(self):
        pass

if __name__ == "__main__":
    unittest.main()
