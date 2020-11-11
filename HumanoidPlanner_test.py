from HumanoidPlanner import HumanoidPlanner
from HumanoidPlanner import create_q_interpolation, create_r_interpolation, apply_angular_velocity_to_quaternion
from HumanoidPlanner import create_constraint_input_array
from pydrake.all import MathematicalProgram, le, ge, eq
import numpy as np
from Atlas import Atlas, load_atlas, set_atlas_initial_pose
import unittest
import pdb

from pydrake.all import MultibodyPlant
from pydrake.autodiffutils import initializeAutoDiff

g = 9.81
g_vec = np.array([0, 0, -g])

mbp_time_step = 1.0e-3
epsilon = 1e-5

def assert_autodiff_array_almost_equal(autodiff_array, float_array):
    float_array = np.array([i.value() for i in autodiff_array])
    np.testing.assert_array_almost_equal(float_array, float_array)

def default_q(N = 1):
    pelvis_orientation = [1., 0., 0., 0.]
    pelvis_position = [0., 0., 0.93845] # Feet just touching ground
    joint_positions = [0.] * Atlas.NUM_ACTUATED_DOF
    return np.array([pelvis_orientation + pelvis_position + joint_positions]*N)

def default_v(N = 1):
    pelvis_rotational_velocity = [0., 0., 0.]
    pelvis_linear_velocity = [0., 0., 0.]
    joint_velocity = [0.] * Atlas.NUM_ACTUATED_DOF
    return np.array([pelvis_rotational_velocity + pelvis_rotational_velocity + joint_velocity] * N)

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
        t = np.array([1.0])
        w_axis = np.array([1., 0., 0.])
        w_mag = np.array([1.0])
        q_new = apply_angular_velocity_to_quaternion(q, w_axis, w_mag, t)
        q_new_expected = np.array([0.877583, 0.479425, 0.0, 0.0])
        np.testing.assert_array_almost_equal(q_new, q_new_expected)

        w_axis = np.array([0., 1., 0.])
        w_mag = np.array([2.0])
        t = np.array([0.5])
        q_new = apply_angular_velocity_to_quaternion(q, w_axis, w_mag, t)
        q_new_expected = np.array([0.877583, 0.0, 0.479425, 0.0])
        np.testing.assert_array_almost_equal(q_new, q_new_expected)

        w_axis = np.array([0., 0., 1.])
        w_mag = 0.5
        t = 2.0
        q_new = apply_angular_velocity_to_quaternion(q, w_axis, w_mag, t)
        q_new_expected = np.array([0.877583, 0.0, 0.0, 0.479425])
        np.testing.assert_array_almost_equal(q_new, q_new_expected)

        w_axis = np.array([0.26726124191242438468, 0.53452248382484876937, 0.80178372573727315405])
        w_mag = 3.74165738677394138558
        t = 1.0
        q_new = apply_angular_velocity_to_quaternion(q, w_axis, w_mag, t)
        q_new_expected = np.array([-0.2955511242573139, 0.25532186031279896, 0.5106437206255979, 0.7659655809383968])
        np.testing.assert_array_almost_equal(q_new, q_new_expected)

    def test_create_constraint_input_array(self):
        prog = MathematicalProgram()
        q = prog.NewContinuousVariables(1, 'q')
        r = prog.NewContinuousVariables(rows=2, cols=3, name='r')
        q_val = 0.5
        r_val = np.arange(6).reshape((2,3))
        '''
        array([[0, 1, 2],
               [3, 4, 5]])
        '''

        constraint = prog.AddConstraint(le(q, r[0,0] + 2*r[1,1]))
        input_array = create_constraint_input_array(constraint, {
            "q": q_val,
            "r": r_val
        })
        expected_input_array = [0.5, 0, 4]
        np.testing.assert_array_almost_equal(input_array, expected_input_array)

        z = prog.NewContinuousVariables(2, 'z')
        z_val = np.array([0.0, 0.5])
        # https://stackoverflow.com/questions/64736910/using-le-or-ge-with-scalar-left-hand-side-creates-unsized-formula-array
        # constraint = prog.AddConstraint(le(z[1], r[0,2] + 2*r[1,0]))
        constraint = prog.AddConstraint(le([z[1]], r[0,2] + 2*r[1,0]))
        input_array = create_constraint_input_array(constraint, {
            "z": z_val,
            "r": r_val
        })
        expected_input_array = [3, 2, 0.5]
        np.testing.assert_array_almost_equal(input_array, expected_input_array)

        a = prog.NewContinuousVariables(rows=2, cols=1, name='a')
        a_val = np.array([[1.0], [2.0]])
        constraint = prog.AddConstraint(eq(a[0], a[1]))
        input_array = create_constraint_input_array(constraint, {
            "a": a_val
        })
        expected_input_array = [1.0, 2.0]
        np.testing.assert_array_almost_equal(input_array, expected_input_array)

class TestHumanoidPlanner(unittest.TestCase):
    def setUp(self):
        self.plant = MultibodyPlant(mbp_time_step)
        load_atlas(self.plant, add_ground=False)
        self.context = self.plant.CreateDefaultContext()
        upright_context = self.plant.CreateDefaultContext()
        set_atlas_initial_pose(self.plant, upright_context)
        self.q_nom = self.plant.GetPositions(upright_context)
        self.planner = HumanoidPlanner(self.plant, Atlas.CONTACTS_PER_FRAME, self.q_nom)

    def create_default_program(self, N=2):
        q_init = self.q_nom.copy()
        q_init[6] = 0.94 # Avoid initializing with ground penetration
        q_final = q_init.copy()
        q_final[4] = 0.0 # x position of pelvis
        q_final[6] = 0.93 # z position of pelvis (to make sure final pose touches ground)
        num_knot_points = N
        max_time = 0.02
        self.planner.create_program(q_init, q_final, num_knot_points, max_time, pelvis_only=True)

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

    def test_eq7c_constraints(self):
        N = 2
        self.create_default_program(N)
        q = default_q(N)
        v = default_v(N)
        h = np.array([[0., 0., 0.]]*N)
        self.assertTrue(self.planner.check_eq7c_constraints(q, v, h))

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
        pelvis_orientation = [0.5934849924416884, 0.2151038891437094, 0.4302077782874188, 0.6453116674311282]
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

    def test_standing_trajectory(self):
        N = 2
        t = 0.5
        num_contacts = 16
        contact_dim = 3*16
        N_d = 4
        q = np.zeros((N, self.plant.num_positions()))
        w_axis = np.zeros((N, 3))
        w_axis[:, 2] = 1.0
        w_mag = np.zeros((N, 1))
        v = np.zeros((N, self.plant.num_velocities()))
        dt = t/N*np.ones((N, 1))
        r = np.zeros((N, 3))
        rd = np.zeros((N, 3))
        rdd = np.zeros((N, 3))
        c = np.zeros((N, contact_dim))
        F = np.zeros((N, contact_dim))
        tau = np.zeros((N, num_contacts))
        h = np.zeros((N, 3))
        hd = np.zeros((N, 3))
        beta = np.zeros((N, num_contacts*N_d))

        ''' Initialize standing pose '''
        for i in range(N):
            q[i][0] = 1.0 # w of quaternion
            q[i][6] = 0.93845 # z of pelvis
            # r[i] = self.planner.calc_r(q[i], v[i])
            # TODO
            # c[i] = 
            F[i] = np.array([0., 0., Atlas.M*g / 16] * 16)

        self.create_default_program()
        ''' Test all constraints satisfied '''
        self.assertTrue(self.planner.check_all_constraints(q, w_axis, w_mag, v, dt, r, rd, rdd, c, F, tau, h, hd, beta))

if __name__ == "__main__":
    unittest.main()
