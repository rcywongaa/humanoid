from HumanoidPlanner import HumanoidPlanner
from HumanoidPlanner import create_q_interpolation, create_r_interpolation, apply_angular_velocity_to_quaternion
from HumanoidPlanner import create_constraint_input_array
from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, ConnectDrakeVisualizer, ConnectContactResultsToDrakeVisualizer, Simulator
from pydrake.all import MathematicalProgram
from pydrake.all import Quaternion, RollPitchYaw
import numpy as np
from Atlas import Atlas, load_atlas, set_atlas_initial_pose, getActuatorIndex, set_null_input
import unittest
import pdb
import time

from pydrake.all import MultibodyPlant
from pydrake.autodiffutils import initializeAutoDiff

g = 9.81
g_vec = np.array([0, 0, -g])

mbp_time_step = 1.0e-3
epsilon = 1e-5

def visualize(q, dt=None):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(mbp_time_step))
    load_atlas(plant, add_ground=True)
    plant_context = plant.CreateDefaultContext()
    ConnectContactResultsToDrakeVisualizer(builder=builder, plant=plant)
    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
    diagram = builder.Build()

    if len(q.shape) == 1:
        q = np.reshape(q, (1, -1))

    for i in range(q.shape[0]):
        print(f"knot point: {i}")
        diagram_context = diagram.CreateDefaultContext()
        plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
        set_null_input(plant, plant_context)

        plant.SetPositions(plant_context, q[i])
        simulator = Simulator(diagram, diagram_context)
        simulator.set_target_realtime_rate(0.0)
        simulator.AdvanceTo(0)
        if not dt is None:
            time.sleep(5/(np.sum(dt))*dt[i])
        else:
            time.sleep(0.5)

def assert_autodiff_array_almost_equal(autodiff_array, float_array):
    float_array = np.array([i.value() for i in autodiff_array])
    np.testing.assert_array_almost_equal(float_array, float_array)

def default_q(N = 0):
    pelvis_orientation = [1., 0., 0., 0.]
    pelvis_position = [0., 0., 0.93845] # Feet just touching ground
    joint_positions = [0.] * Atlas.NUM_ACTUATED_DOF
    if N == 0:
        return np.array(pelvis_orientation + pelvis_position + joint_positions)
    else:
        return np.array([pelvis_orientation + pelvis_position + joint_positions]*N)

def default_v(N = 0):
    pelvis_rotational_velocity = [0., 0., 0.]
    pelvis_linear_velocity = [0., 0., 0.]
    joint_velocity = [0.] * Atlas.NUM_ACTUATED_DOF
    if N == 0:
        return np.array(pelvis_rotational_velocity + pelvis_rotational_velocity + joint_velocity)
    else:
        return np.array([pelvis_rotational_velocity + pelvis_rotational_velocity + joint_velocity] * N)

def default_c(N = 0):
    if N == 0:
        return np.zeros(3*16)
    else:
        return np.zeros((N, 3*16))

def default_slack(N=0):
    if N == 0:
        return np.zeros(1)
    else:
        return np.zeros((N, 1))

class TestHumanoidPlannerStandalone(unittest.TestCase):
    def test_create_q_interpolation(self):
        plant = MultibodyPlant(mbp_time_step)
        load_atlas(plant, add_ground=False)
        context = plant.CreateDefaultContext()
        self.skipTest("Unimplemented")

    def test_create_r_interpolation(self):
        self.skipTest("Unimplemented")

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

        self.num_contacts = 16
        self.contact_dim = 3*16
        self.N_d = 4

    def create_default_program(self, N=2):
        q_init = self.q_nom.copy()
        q_init[6] = 0.94 # Avoid initializing with ground penetration
        q_final = q_init.copy()
        q_final[4] = 0.0 # x position of pelvis
        q_final[6] = 0.93 # z position of pelvis (to make sure final pose touches ground)
        num_knot_points = N
        max_time = 0.02
        # TODO: Use create_minimal_program instead
        self.planner.create_full_program(q_init, q_final, num_knot_points, max_time, pelvis_only=True)

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

    def test_reshape_tauj(self):
        tau_k = [i for i in range(16)]
        tau_j = self.planner.reshape_tauj(tau_k)
        tau_j_expected = np.array([[0.0, 0.0, i] for i in range(16)])
        np.testing.assert_array_almost_equal(tau_j, tau_j_expected)

    def test_get_contact_position(self):
        self.skipTest("Unimplemented")

    # def test_get_contact_positions_z(self):
        # pelvis_orientation = [1., 0., 0., 0.]
        # pelvis_position = [0., 0., 0.93845]
        # joint_positions = [0.] * Atlas.NUM_ACTUATED_DOF
        # q = np.array(pelvis_orientation + pelvis_position + joint_positions)
        # pelvis_rotational_velocity = [0., 0., 0.]
        # pelvis_linear_velocity = [0., 0., 0.]
        # joint_velocity = [0.] * Atlas.NUM_ACTUATED_DOF
        # v = np.array(pelvis_rotational_velocity + pelvis_rotational_velocity + joint_velocity)
        # contact_positions_z = self.planner.get_contact_positions_z(q, v)
        # expected_contact_positions_z = [0.] * Atlas.NUM_CONTACTS
        # np.testing.assert_allclose(contact_positions_z, expected_contact_positions_z, atol=epsilon)

        # pelvis_orientation = [1., 0., 0., 0.]
        # pelvis_position = [0., 0., 1.03845]
        # joint_positions = [0.] * Atlas.NUM_ACTUATED_DOF
        # q = np.array(pelvis_orientation + pelvis_position + joint_positions)
        # pelvis_rotational_velocity = [0., 0., 0.]
        # pelvis_linear_velocity = [0., 0., 0.]
        # joint_velocity = [0.] * Atlas.NUM_ACTUATED_DOF
        # v = np.array(pelvis_rotational_velocity + pelvis_rotational_velocity + joint_velocity)
        # contact_positions_z = self.planner.get_contact_positions_z(q, v)
        # expected_contact_positions_z = [0.1] * Atlas.NUM_CONTACTS
        # np.testing.assert_allclose(contact_positions_z, expected_contact_positions_z, atol=epsilon)

    def test_eq7a_constraints(self):
        N = 2
        self.create_default_program(N)
        F = np.zeros((N, self.contact_dim))
        rdd = np.zeros((N, 3))
        rdd[0][2] = -g
        rdd[1][2] = -g
        self.assertTrue(self.planner.check_eq7a_constraints(F, rdd))

        F[0][2::3] = [Atlas.M*g/self.num_contacts]*self.num_contacts
        F[1][2::3] = [Atlas.M*g/self.num_contacts]*self.num_contacts
        rdd[0][2] = 0.0
        rdd[1][2] = 0.0
        self.assertTrue(self.planner.check_eq7a_constraints(F, rdd))

    def test_eq7b_constraints(self):
        self.skipTest("Unimplemented")

    def test_eq7c(self):
        q = default_q()
        v = default_v()
        h = np.array([0., 0., 0.])
        q_v_h = np.concatenate([q, v, h])
        np.testing.assert_allclose(self.planner.eq7c(q_v_h), 0.)

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
        w_axis = [0., 0., 1.]
        w_mag = [0.0]
        v = default_v()
        dt = [0.5]
        q_qprev_waxis_wmag_v_dt = np.concatenate([q, qprev, w_axis, w_mag, v, dt])
        np.testing.assert_allclose(self.planner.eq7d(q_qprev_waxis_wmag_v_dt), 0.)

        w_axis = [0.26726124191242438468, 0.53452248382484876937, 0.80178372573727315405]
        w_mag = [3.74165738677394138558]
        pelvis_rotational_velocity = [1., 2., 3.]
        pelvis_linear_velocity = [1., 2., 3.]
        joint_velocity = [i for i in range(Atlas.NUM_ACTUATED_DOF)]
        v = np.array(pelvis_rotational_velocity + pelvis_linear_velocity + joint_velocity)
        pelvis_orientation = [0.5934849924416884, 0.2151038891437094, 0.4302077782874188, 0.6453116674311282]
        pelvis_position = [0.5, 1., 1.5+0.93845]
        joint_positions = [i*0.5 for i in range(Atlas.NUM_ACTUATED_DOF)]
        q = np.array(pelvis_orientation + pelvis_position + joint_positions)
        q_qprev_waxis_wmag_v_dt = np.concatenate([q, qprev, w_axis, w_mag, v, dt])
        np.testing.assert_allclose(self.planner.eq7d(q_qprev_waxis_wmag_v_dt), 0., atol=epsilon)

    def test_eq7d_constraints(self):
        N = 2
        self.create_default_program(N)
        q = default_q(N)
        w_axis = np.zeros((N, 3))
        w_mag = np.zeros((N,1))
        dt = np.zeros((2,1))
        w_axis[0] = [0., 0., 1.]
        w_mag[0] = 0

        pelvis_orientation = [0.5934849924416884, 0.2151038891437094, 0.4302077782874188, 0.6453116674311282]
        pelvis_position = [0.5, 1., 1.5+0.93845]
        joint_positions = [i*0.5 for i in range(Atlas.NUM_ACTUATED_DOF)]
        q[1] = np.array(pelvis_orientation + pelvis_position + joint_positions)
        w_axis[1] = [0.26726124191242438468, 0.53452248382484876937, 0.80178372573727315405]
        w_mag[1] = 3.74165738677394138558
        v = default_v(N)
        pelvis_rotational_velocity = [1., 2., 3.]
        pelvis_linear_velocity = [1., 2., 3.]
        joint_velocity = [i for i in range(Atlas.NUM_ACTUATED_DOF)]
        v[1] = np.array(pelvis_rotational_velocity + pelvis_linear_velocity + joint_velocity)
        dt[1] = 0.5
        self.assertTrue(self.planner.check_eq7d_constraints(q, w_axis, w_mag, v, dt))

    def test_eq7e_constraints(self):
        N = 2
        self.create_default_program(N)
        h = np.zeros((N, 3))
        hd = np.zeros((N, 3))
        dt = np.zeros((N, 1))
        dt[1] = 1.0
        self.assertTrue(self.planner.check_eq7e_constraints(h, hd, dt))

        hd[1] = [1.0, -2.0, 3.0]
        h[1] = [0.5, -1.0, 1.5]
        dt[1] = 0.5
        self.assertTrue(self.planner.check_eq7e_constraints(h, hd, dt))

    def test_eq7f_constraints(self):
        N = 2
        self.create_default_program(N)
        r = np.zeros((N, 3))
        rd = np.zeros((N, 3))
        dt = np.zeros((N, 1))
        dt[1] = 1.0
        self.assertTrue(self.planner.check_eq7f_constraints(r, rd, dt))

        rd[1] = [1.0, -2.0, 3.0]
        r[0] = [0.0, 0.0, 0.0]
        r[1] = [0.25, -0.5, 0.75]
        dt[1] = 0.5
        self.assertTrue(self.planner.check_eq7f_constraints(r, rd, dt))

    def test_eq7g_constraints(self):
        N = 2
        self.create_default_program(N)
        rd = np.zeros((N, 3))
        rdd = np.zeros((N, 3))
        dt = np.zeros((N, 1))
        dt[1] = 1.0
        self.assertTrue(self.planner.check_eq7g_constraints(rd, rdd, dt))

        rdd[1] = [1.0, -2.0, 3.0]
        rd[1] = [0.5, -1.0, 1.5]
        dt[1] = 0.5
        self.assertTrue(self.planner.check_eq7g_constraints(rd, rdd, dt))

    def test_eq7h(self):
        self.skipTest("Unimplemented")

    def test_eq7h_constraints(self):
        self.skipTest("Unimplemented")

    def test_eq7i(self):
        self.skipTest("Unimplemented")

    def test_eq7i_constraints(self):
        self.skipTest("Unimplemented")

    def test_eq7j_constraints(self):
        N = 2
        self.create_default_program(N)
        c = np.zeros((N, self.contact_dim))
        self.assertTrue(self.planner.check_eq7j_constraints(c))

        c[0] = [0.1, -9.9, 5.0] * self.num_contacts
        c[1] = [-9.9, 0.1, 0.0] * self.num_contacts
        self.assertTrue(self.planner.check_eq7j_constraints(c))

        c[0] = [0.1, -9.9, -0.5] * self.num_contacts
        c[1] = [-9.9, 0.1, 0.01] * self.num_contacts
        self.assertFalse(self.planner.check_eq7j_constraints(c))

    def test_eq7k_admissable_posture_constraints(self):
        N = 2
        self.create_default_program(N)
        q = default_q(N)
        self.assertTrue(self.planner.check_eq7k_admissable_posture_constraints(q))

        q[0, Atlas.FLOATING_BASE_QUAT_DOF + getActuatorIndex(self.plant, "l_leg_kny")] = 0
        q[1, Atlas.FLOATING_BASE_QUAT_DOF + getActuatorIndex(self.plant, "l_leg_kny")] = 2.35637
        self.assertTrue(self.planner.check_eq7k_admissable_posture_constraints(q))

        q[0, Atlas.FLOATING_BASE_QUAT_DOF + getActuatorIndex(self.plant, "l_leg_kny")] = -0.1
        q[1, Atlas.FLOATING_BASE_QUAT_DOF + getActuatorIndex(self.plant, "l_leg_kny")] = 2.4
        self.assertFalse(self.planner.check_eq7k_admissable_posture_constraints(q))

    def test_eq7k_joint_velocity_constraints(self):
        N = 2
        self.create_default_program(N)
        v = default_v(N)
        self.assertTrue(self.planner.check_eq7k_joint_velocity_constraints(v))

        v[0, Atlas.FLOATING_BASE_DOF + getActuatorIndex(self.plant, "r_leg_kny")] = -12
        v[1, Atlas.FLOATING_BASE_DOF + getActuatorIndex(self.plant, "r_leg_kny")] = 12
        self.assertTrue(self.planner.check_eq7k_joint_velocity_constraints(v))

        v[0, Atlas.FLOATING_BASE_DOF + getActuatorIndex(self.plant, "r_leg_kny")] = -12.1
        v[1, Atlas.FLOATING_BASE_DOF + getActuatorIndex(self.plant, "r_leg_kny")] = 12.1
        self.assertFalse(self.planner.check_eq7k_joint_velocity_constraints(v))

    def test_eq7k_friction_cone_constraints(self):
        N = 2
        self.create_default_program(N)
        F = np.zeros((N, self.contact_dim))
        beta = np.zeros((N, self.num_contacts*self.N_d))
        self.assertTrue(self.planner.check_eq7k_friction_cone_constraints(F, beta))

        F[0, 3] = 1.0 # 2nd contact x
        F[0, 4] = 2.0 # 2nd contact y
        F[0, 5] = 3.0 # 2nd contact z
        beta[0, 4] = 1.0 # 2nd contact [1.0, 0.0, 1.0] component
        beta[0, 5] = 0.0 # 2nd contact [-1.0, 0.0, 1.0] component
        beta[0, 6] = 2.0 # 2nd contact [0.0, 1.0, 1.0] component
        beta[0, 7] = 0.0 # 2nd contact [0.0, -1.0, 1.0] component
        self.assertTrue(self.planner.check_eq7k_friction_cone_constraints(F, beta))

    def test_eq7k_beta_positive_constraints(self):
        N = 2
        self.create_default_program(N)
        beta = np.zeros((N, self.num_contacts*self.N_d))
        self.assertTrue(self.planner.check_eq7k_beta_positive_constraints(beta))

        beta[1][10] = -0.1
        self.assertFalse(self.planner.check_eq7k_beta_positive_constraints(beta))

        beta[1][10] = 0.1
        self.assertTrue(self.planner.check_eq7k_beta_positive_constraints(beta))

    def test_eq7k_torque_constraints(self):
        N = 2
        self.create_default_program(N)
        beta = np.zeros((N, self.num_contacts*self.N_d))
        tau = np.zeros((N, self.num_contacts))
        self.assertTrue(self.planner.check_eq7k_torque_constraints(tau, beta))

        tau[0][3] = 1.0
        self.assertFalse(self.planner.check_eq7k_torque_constraints(tau, beta))

        tau[0][3] = 0.5
        beta[0][4*3] = 0.1
        beta[0][4*3+1] = 0.2
        beta[0][4*3+2] = 0.3
        beta[0][4*3+3] = 0.4
        self.assertTrue(self.planner.check_eq7k_torque_constraints(tau, beta))

    def test_eq8a_constraints(self):
        N = 2
        self.create_default_program(N)
        F = np.zeros((N, self.contact_dim))
        c = default_c(N)
        slack = default_slack(N)
        self.assertTrue(self.planner.check_eq8a_constraints(F, c, slack))

        c[1][11*3+2] = 0.0
        F[1][11*3+2] = 10.0
        self.assertTrue(self.planner.check_eq8a_constraints(F, c, slack))

        c[1][10*3+2] = 1.0
        F[1][11*3+2] = 10.0
        self.assertTrue(self.planner.check_eq8a_constraints(F, c, slack))

        c[1][11*3+2] = 1.0
        F[1][11*3+2] = 10.0
        self.assertFalse(self.planner.check_eq8a_constraints(F, c, slack))

    def test_eq8b_constraints(self):
        N = 2
        self.create_default_program(N)
        c = default_c(N)
        tau = np.zeros((N, self.num_contacts))
        slack = default_slack(N)
        self.assertTrue(self.planner.check_eq8b_constraints(tau, c, slack))

        c[0][8*3+2] = 0.0
        tau[0][8] = -1.0
        self.assertTrue(self.planner.check_eq8b_constraints(tau, c, slack))

        c[0][9*3+2] = 0.0
        tau[0][8] = 1.0
        self.assertTrue(self.planner.check_eq8b_constraints(tau, c, slack))

        c[0][8*3+2] = 1.0
        tau[0][8] = 1.0
        self.assertFalse(self.planner.check_eq8b_constraints(tau, c, slack))

    def test_eq8c_contact_force_constraints(self):
        N = 2
        self.create_default_program(N)
        F = np.zeros((N, self.contact_dim))
        self.assertTrue(self.planner.check_eq8c_contact_force_constraints(F))

        F[0][3] = -0.1 # 2nd contact x
        F[0][4] = -0.1 # 2nd contact y
        F[0][5] = 0.1 # 2nd contact z
        self.assertTrue(self.planner.check_eq8c_contact_force_constraints(F))

        F[0][3] = 0.1 # 2nd contact x
        F[0][4] = 0.1 # 2nd contact y
        F[0][5] = -0.1 # 2nd contact z
        self.assertFalse(self.planner.check_eq8c_contact_force_constraints(F))

    def test_eq8c_contact_distance_constraint(self):
        N = 2
        self.create_default_program(N)
        c = default_c(N)
        self.assertTrue(self.planner.check_eq8c_contact_distance_constraints(c))

        c[1][3*2] = -1.0
        c[1][3*2+1] = -1.0
        c[1][3*2+2] = 0.1
        self.assertTrue(self.planner.check_eq8c_contact_distance_constraints(c))

        c[0][3*2] = -1.0
        c[0][3*2+1] = -1.0
        c[0][3*2+2] = -0.1
        self.assertFalse(self.planner.check_eq8c_contact_distance_constraints(c))

    def test_eq9a_lhs(self):
        self.skipTest("Unimplemented")

    def test_eq9a_constraints(self):
        N = 2
        self.create_default_program(N)
        F = np.zeros((N, self.contact_dim))
        c = np.zeros((N, self.contact_dim))
        slack = default_slack(N)
        self.assertTrue(self.planner.check_eq9a_constraints(F, c, slack))

        ''' Allow contact point to move if contact force on another contact '''
        F = np.zeros((N, self.contact_dim))
        c = np.zeros((N, self.contact_dim))
        F[1][3*15 + 2] = 0.2 # 16th contact z
        c[0, 3*14] = 0.1 # 15th contact x
        c[1, 3*14] = 0.2 # 15th contact x
        self.assertTrue(self.planner.check_eq9a_constraints(F, c, slack))

        ''' Contact point should not move when applying contact force '''
        F = np.zeros((N, self.contact_dim))
        c = np.zeros((N, self.contact_dim))
        F[1][3*15 + 2] = 0.2 # 16th contact z
        c[0, 3*15] = 0.1 # 16th contact x
        c[1, 3*15] = 0.2 # 16th contact x
        self.assertFalse(self.planner.check_eq9a_constraints(F, c, slack))

        ''' Allow contact points to move if no contact force '''
        F = np.zeros((N, self.contact_dim))
        c = np.zeros((N, self.contact_dim))
        F[1][3*15 + 2] = 0.0 # 16th contact z
        c[0, 3*15] = 0.1 # 16th contact x
        c[1, 3*15] = 0.2 # 16th contact x
        self.assertTrue(self.planner.check_eq9a_constraints(F, c, slack))

        ''' This should not check for y axis slipping '''
        F = np.zeros((N, self.contact_dim))
        c = np.zeros((N, self.contact_dim))
        F[1][3*15 + 2] = 0.2 # 16th contact z
        c[0, 3*15+1] = 0.1 # 16th contact y
        c[1, 3*15+1] = 0.2 # 16th contact y
        self.assertTrue(self.planner.check_eq9a_constraints(F, c, slack))

    def test_eq9b_lhs(self):
        self.skipTest("Unimplemented")

    def test_eq9b_constraints(self):
        N = 2
        self.create_default_program(N)
        F = np.zeros((N, self.contact_dim))
        c = np.zeros((N, self.contact_dim))
        slack = default_slack(N)
        self.assertTrue(self.planner.check_eq9b_constraints(F, c, slack))

        ''' Allow contact point to move if contact force on another contact '''
        F = np.zeros((N, self.contact_dim))
        c = np.zeros((N, self.contact_dim))
        F[1][3*15 + 2] = 0.2 # 16th contact z
        c[0, 3*14+1] = 0.1 # 15th contact y
        c[1, 3*14+1] = 0.2 # 15th contact y
        self.assertTrue(self.planner.check_eq9b_constraints(F, c, slack))

        ''' Contact point should not move when applying contact force '''
        F = np.zeros((N, self.contact_dim))
        c = np.zeros((N, self.contact_dim))
        F[1][3*15 + 2] = 0.2 # 16th contact z
        c[0, 3*15+1] = 0.1 # 16th contact y
        c[1, 3*15+1] = 0.2 # 16th contact y
        self.assertFalse(self.planner.check_eq9b_constraints(F, c, slack))

        ''' Allow contact points to move if no contact force '''
        F = np.zeros((N, self.contact_dim))
        c = np.zeros((N, self.contact_dim))
        F[1][3*15 + 2] = 0.0 # 16th contact x
        c[0, 3*15+1] = 0.1 # 16th contact y
        c[1, 3*15+1] = 0.2 # 16th contact y
        self.assertTrue(self.planner.check_eq9b_constraints(F, c, slack))

        ''' This should not check for x axis slipping '''
        F = np.zeros((N, self.contact_dim))
        c = np.zeros((N, self.contact_dim))
        F[1][3*15 + 2] = 0.2 # 16th contact z
        c[0, 3*15] = 0.1 # 16th contact x
        c[1, 3*15] = 0.2 # 16th contact x
        self.assertTrue(self.planner.check_eq9b_constraints(F, c, slack))

    def test_pose_error_cost(self):
        self.skipTest("Unimplemented")

    def test_standing_trajectory(self):
        self.skipTest("Unimplemented")
        return
        N = 2
        t = 0.5
        q = np.zeros((N, self.plant.num_positions()))
        w_axis = np.zeros((N, 3))
        w_axis[:, 2] = 1.0
        w_mag = np.zeros((N, 1))
        v = np.zeros((N, self.plant.num_velocities()))
        dt = t/N*np.ones((N, 1))
        r = np.zeros((N, 3))
        rd = np.zeros((N, 3))
        rdd = np.zeros((N, 3))
        c = np.zeros((N, self.contact_dim))
        F = np.zeros((N, self.contact_dim))
        tau = np.zeros((N, self.num_contacts))
        h = np.zeros((N, 3))
        hd = np.zeros((N, 3))
        beta = np.zeros((N, self.num_contacts*self.N_d))

        ''' Initialize standing pose '''
        for i in range(N):
            q[i][0] = 1.0 # w of quaternion
            q[i][6] = 0.93845 # z of pelvis
            # r[i] = self.planner.calc_r(q[i], v[i])
            # TODO
            # c[i] = ...
            F[i] = np.array([0., 0., Atlas.M*g / 16] * 16)

        self.create_default_program()
        ''' Test all constraints satisfied '''
        self.assertTrue(self.planner.check_all_constraints(q, w_axis, w_mag, v, dt, r, rd, rdd, c, F, tau, h, hd, beta))

    def test_minimal(self):
        self.planner.create_minimal_program(50, 1)
        is_success, sol = self.planner.solve()
        self.assertTrue(is_success)

    def test_0th_order(self):
        N = 50
        self.planner.create_minimal_program(N, 1)
        q_init = default_q()
        q_init[6] = 1.5 # z position of pelvis
        q_final = default_q()
        q_final[0:4] = Quaternion(RollPitchYaw([2*np.pi, np.pi, np.pi/2]).ToRotationMatrix().matrix()).wxyz()
        q_final[4] = 1.0 # x position of pelvis
        q_final[6] = 2.0 # z position of pelvis
        q_final[15] = np.pi/4 # right hip joint swing back
        self.planner.add_0th_order_constraints(q_init, q_final, False)
        is_success, sol = self.planner.solve(self.planner.create_initial_guess())
        self.assertTrue(is_success)
        visualize(sol.q)

    def test_1st_order(self):
        N = 20
        self.planner.create_minimal_program(N, 0.5)
        q_init = default_q()
        q_init[6] = 1.5 # z position of pelvis
        q_final = default_q()
        q_final[0:4] = Quaternion(RollPitchYaw([2*np.pi, np.pi, np.pi/2]).ToRotationMatrix().matrix()).wxyz()
        q_final[4] = 1.0 # x position of pelvis
        q_final[6] = 2.0 # z position of pelvis
        q_final[15] = np.pi/4 # right hip joint swing back
        self.planner.add_0th_order_constraints(q_init, q_final, False)
        self.planner.add_1st_order_constraints()
        is_success, sol = self.planner.solve(self.planner.create_initial_guess())
        self.assertTrue(is_success)
        visualize(sol.q)

    def test_2nd_order(self):
        N = 20
        self.planner.create_minimal_program(N, 1.0)
        q_init = default_q()
        q_init[6] = 1.0 # z position of pelvis
        q_final = default_q()
        q_final[0:4] = Quaternion(RollPitchYaw([0.0, 0.0, np.pi/6]).ToRotationMatrix().matrix()).wxyz()
        q_final[4] = 1.0 # x position of pelvis
        q_final[6] = 1.5 # z position of pelvis
        q_final[15] = np.pi/4 # right hip joint swing back
        self.planner.add_0th_order_constraints(q_init, q_final, False)
        self.planner.add_1st_order_constraints()
        self.planner.add_2nd_order_constraints()
        is_success, sol = self.planner.solve(self.planner.create_initial_guess())
        self.assertTrue(is_success)
        visualize(sol.q)

    def test_complementarity_constraints(self):
        N = 20
        self.planner.create_minimal_program(N, 1.0)
        q_init = default_q()
        q_init[6] = 1.0 # z position of pelvis
        q_final = default_q()
        q_final[0:4] = Quaternion(RollPitchYaw([0.0, 0.0, 0.0]).ToRotationMatrix().matrix()).wxyz()
        q_final[4] = 1.0 # x position of pelvis
        q_final[6] = 1.0 # z position of pelvis
        self.planner.add_0th_order_constraints(q_init, q_final, False)
        self.planner.add_1st_order_constraints()
        self.planner.add_2nd_order_constraints()
        is_success, sol = self.planner.solve(self.planner.create_initial_guess())
        if is_success:
            print("First pass solution found!")
            self.planner.add_eq8a_constraints()
            is_success, sol = self.planner.solve(self.planner.create_guess(sol))
        # self.planner.add_eq10_cost()
        self.assertTrue(is_success)
        visualize(sol.q)
        pdb.set_trace()

if __name__ == "__main__":
    unittest.main()
