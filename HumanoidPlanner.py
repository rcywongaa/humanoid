#!/usr/bin/python3

'''
This implements the paper
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics
by Hongkai Dai, Andrés Valenzuela and Russ Tedrake
'''

from Atlas import load_atlas, set_atlas_initial_pose
from Atlas import getSortedJointLimits, getActuatorIndex, getActuatorIndices, getJointValues
from Atlas import Atlas
from pydrake.all import Quaternion, AddUnitQuaternionConstraintOnPlant
from pydrake.all import Multiplexer
from pydrake.all import PiecewisePolynomial, PiecewiseTrajectory, PiecewiseQuaternionSlerp, TrajectorySource
from pydrake.all import ConnectDrakeVisualizer, ConnectContactResultsToDrakeVisualizer, Simulator
from pydrake.all import DiagramBuilder, MultibodyPlant, AddMultibodyPlantSceneGraph, BasicVector, LeafSystem
from pydrake.all import MathematicalProgram, Solve, eq, le, ge, SolverOptions
# from pydrake.all import IpoptSolver
from pydrake.all import SnoptSolver
from pydrake.all import Quaternion_, AutoDiffXd
from HumanoidController import HumanoidController
import numpy as np
import time
import pdb
import pickle
from collections.abc import Iterable

mbp_time_step = 1.0e-3
N_f = 3 # contact force dimension
mu = 1.0 # Coefficient of friction
friction_torque_coefficient = 0.5
epsilon = 1e-9
quaternion_epsilon = 1e-5
PLAYBACK_ONLY = False
ENABLE_COMPLEMENTARITY_CONSTRAINTS = True
MAX_GROUND_PENETRATION = 0.0
MAX_JOINT_ACCELERATION = 20.0
g = np.array([0, 0, -Atlas.g])
MIN_TIMESTEP = 0.001
MAX_TIMESTEP = 0.1
'''
Slack for the complementary constraints
Same value used in drake/multibody/optimization/static_equilibrium_problem.cc
'''
# slack = 1e-3

def create_q_interpolation(plant, context, q_traj, v_traj, dt_traj):
    t_traj = np.cumsum(dt_traj)
    quaternions = [Quaternion(q[0:4] / np.linalg.norm(q[0:4])) for q in q_traj]
    quaternion_poly = PiecewiseQuaternionSlerp(t_traj, quaternions)
    position_poly = PiecewisePolynomial.FirstOrderHold(t_traj, q_traj[:, 4:].T)
    return quaternion_poly, position_poly
    # qd_poly = q_poly.derivative()
    # qdd_poly = qd_poly.derivative()
    # return q_poly, v_poly, vd_poly

def create_r_interpolation(r_traj, rd_traj, rdd_traj, dt_traj):
    r_poly = PiecewisePolynomial()
    t = 0.0
    for i in range(len(dt_traj)-1):
        # CubicHermite assumes samples are column vectors
        r = np.array([r_traj[i]]).T
        rd = np.array([rd_traj[i]]).T
        rdd = np.array([rdd_traj[i]]).T
        dt = dt_traj[i+1]
        r_next = np.array([r_traj[i+1]]).T
        rd_next = np.array([rd_traj[i+1]]).T
        rdd_next = np.array([rdd_traj[i+1]]).T
        r_poly.ConcatenateInTime(
                PiecewisePolynomial.CubicHermite(
                    breaks=[t, t+dt],
                    samples=np.hstack([r, r_next]),
                    samples_dot=np.hstack([rd, rd_next])))
        t += dt
    rd_poly = r_poly.derivative()
    rdd_poly = rd_poly.derivative()
    return r_poly, rd_poly, rdd_poly

def quat_multiply(q0, q1):
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                   x1*w0 + y1*z0 - z1*y0 + w1*x0,
                  -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                   x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=q0.dtype)

def apply_angular_velocity_to_quaternion(q, w_axis, w_mag, t):
    delta_q = np.hstack([np.cos(w_mag* t/2.0), w_axis*np.sin(w_mag* t/2.0)])
    return  quat_multiply(q, delta_q)

def get_index_of_variable(variables, variable_name):
    return [str(element) for element in variables].index(variable_name)

def create_constraint_input_array(constraint, name_value_map):
    processed_name_value_map = name_value_map.copy()
    ret = np.zeros(len(constraint.variables()))
    # Fill array with NaN values first
    ret.fill(np.nan)
    for name, value in name_value_map.items():
        if isinstance(value, Iterable):
            # pydrake treats 2D arrays with 1 col / 1 row as 1D array
            if any(np.array(value.shape) == 1):
                value = value.flatten()
            # Expand vectors into individual entries
            it = np.nditer(value, flags=['multi_index', 'refs_ok'])
            while not it.finished:
                if len(it.multi_index) == 1:
                    # Convert x(1,) to x(1)
                    element_name = name + f"({it.multi_index[0]})"
                else:
                    element_name = name + str(it.multi_index).replace(" ","")
                processed_name_value_map[element_name] = it.value
                it.iternext()
            del processed_name_value_map[name]
        else:
            # Rename scalars from 'x' to 'x(0)'
            element_name = name + "(0)"
            processed_name_value_map[element_name] = value
            del processed_name_value_map[name]

    for name, value in processed_name_value_map.items():
        try:
            ret[get_index_of_variable(constraint.variables(), name)] = value
        except ValueError:
            pass
    # Make sure all values are filled
    nan_idx = np.argwhere(np.isnan(ret))
    if nan_idx.size != 0:
        raise Exception(f"{constraint.variables()[nan_idx[0,0]]} missing!")
    return ret

def check_constraint(constraint, input_map):
    input_array = create_constraint_input_array(constraint, input_map)
    if constraint.evaluator().CheckSatisfied(input_array):
        return True
    else:
        print(f"{constraint.evaluator().get_description()} violated!")
        return False

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def check_constraints(constraints, input_map):
    for constraint in flatten(constraints):
        if not check_constraint(constraint, input_map):
            return False
    return True

class HumanoidPlanner:
    def __init__(self, plant_float, contacts_per_frame, q_nom):
        self.plant_float = plant_float
        self.context_float = plant_float.CreateDefaultContext()
        self.plant_autodiff = self.plant_float.ToAutoDiffXd()
        self.context_autodiff = self.plant_autodiff.CreateDefaultContext()
        self.q_nom = q_nom

        self.sorted_joint_position_lower_limits = np.array([entry[1].lower for entry in getSortedJointLimits(self.plant_float)])
        self.sorted_joint_position_upper_limits = np.array([entry[1].upper for entry in getSortedJointLimits(self.plant_float)])
        self.sorted_joint_velocity_limits = np.array([entry[1].velocity for entry in getSortedJointLimits(self.plant_float)])

        self.contacts_per_frame = contacts_per_frame
        self.num_contacts = sum([contact_points.shape[1] for contact_points in contacts_per_frame.values()])
        self.contact_dim = 3*self.num_contacts

        self.N_d = 4 # friction cone approximated as a i-pyramid
        n = np.array([
            [0],
            [0],
            [1.0]])
        d = np.array([
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0]])
        # Equivalent to v in HumanoidController.py
        self.friction_cone_components = np.zeros((self.N_d, self.num_contacts, N_f))
        for i in range(self.N_d):
            for j in range(self.num_contacts):
                self.friction_cone_components[i,j] = (n+mu*d)[:,i]

    def getPlantAndContext(self, q, v):
        assert(q.dtype == v.dtype)
        if q.dtype == np.object:
            self.plant_autodiff.SetPositions(self.context_autodiff, q)
            self.plant_autodiff.SetVelocities(self.context_autodiff, v)
            return self.plant_autodiff, self.context_autodiff
        else:
            self.plant_float.SetPositions(self.context_float, q)
            self.plant_float.SetVelocities(self.context_float, v)
            return self.plant_float, self.context_float

    '''
    Reshapes vector of tau[k] of shape [num_contacts, 1]
    into an np.array of shape [num_contacts, 3] where first 2 rows are zeros
    since we only care about tau in the z direction
    '''
    def reshape_tauj(self, tau_k):
        return np.hstack([np.zeros((self.num_contacts, 2)), np.reshape(tau_k, (self.num_contacts, 1))])

    '''
    Reshapes np.array of shape [3*num_contacts] to np.array of shape [num_contacts, 3]
    '''
    def reshape_3d_contact(self, v):
        return np.reshape(v, (self.num_contacts, 3))

    '''
    Returns contact positions in the shape [3, num_contacts]
    '''
    def get_contact_positions(self, q, v):
        plant, context = self.getPlantAndContext(q, v)
        contact_positions_per_frame = []
        for frame, contacts in self.contacts_per_frame.items():
            contact_positions = plant.CalcPointsPositions(
                context, plant.GetFrameByName(frame),
                contacts, plant.world_frame())
            contact_positions_per_frame.append(contact_positions)
        return np.concatenate(contact_positions_per_frame, axis=1)

    ''' Use c variable instead '''
    # def get_contact_positions_z(self, q, v):
        # return self.get_contact_positions(q, v)[2,:]

    # https://stackoverflow.com/questions/63454077/how-to-obtain-centroidal-momentum-matrix/63456202#63456202
    def calc_h(self, q, v):
        plant, context = self.getPlantAndContext(q, v)
        return plant.CalcSpatialMomentumInWorldAboutPoint(context, plant.CalcCenterOfMassPosition(context)).rotational()

    def calc_r(self, q, v):
        plant, context = self.getPlantAndContext(q, v)
        return plant.CalcCenterOfMassPosition(context)

    def pose_error_cost(self, q_v_dt):
        q, v, dt = np.split(q_v_dt, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        plant, context = self.getPlantAndContext(q, v)
        Q_q = 1.0 * np.identity(plant.num_velocities())
        q_err = plant.MapQDotToVelocity(context, q-self.q_nom)
        return (dt*(q_err.dot(Q_q).dot(q_err)))[0] # AddCost requires cost function to return scalar, not array

    def add_constraint(self, func, *args, ub =[0], lb =[0]):
        arg_list = list(args)
        for i in range(len(arg_list)):
            if not isinstance(arg_list[i], Iterable):
                arg_list[i] = [arg_list[i]]

        arg_sizes = list(map(lambda arg : len(arg), arg_list))
        arg_splits = np.cumsum(arg_sizes)[:-1]

        def wrapped_func(concatenated_args, arg_splits):
            return func(*(np.split(concatenated_args, arg_splits)))

        return self.prog.AddConstraint(
                lambda concatenated_args, arg_splits=arg_splits : wrapped_func(concatenated_args, arg_splits), # Must return a vector
                lb=lb,
                ub=ub,
                vars=np.concatenate(arg_list))

    def add_eq7a_constraints(self):
        F = self.F
        rdd = self.rdd
        self.eq7a_constraints = []
        for k in range(self.N):
            Fj = self.reshape_3d_contact(F[k])
            constraint = self.prog.AddLinearConstraint(
                    eq(Atlas.M*rdd[k], np.sum(Fj, axis=0) + Atlas.M*g))
            constraint.evaluator().set_description(f"Eq(7a)[{k}]")
            self.eq7a_constraints.append(constraint)

    def check_eq7a_constraints(self, F, rdd):
        return check_constraints(self.eq7a_constraints, {
            "F": F,
            "rdd": rdd
        })

    def add_eq7b_constraints(self):
        F = self.F
        c = self.c
        tau = self.tau
        hd = self.hd
        r = self.r
        self.eq7b_constraints = []
        for k in range(self.N):
            Fj = self.reshape_3d_contact(F[k])
            cj = self.reshape_3d_contact(c[k])
            tauj = self.reshape_tauj(self.tau[k])
            constraint = self.prog.AddConstraint(
                    eq(hd[k], np.sum(np.cross(cj - r[k], Fj) + tauj, axis=0)))
            constraint.evaluator().set_description(f"Eq(7b)[{k}]")
            self.eq7b_constraints.append(constraint)

    def check_eq7b_constraints(self, F, c, tau, hd, r):
        return check_constraints(self.eq7b_constraints, {
            "F": F,
            "c": c,
            "tau": tau,
            "hd": hd,
            "r": r
        })

    def eq7c(self, q_v_h):
        q, v, h = np.split(q_v_h, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        return self.calc_h(q, v) - h

    def add_eq7c_constraints(self):
        q = self.q
        v = self.v
        h = self.h
        self.eq7c_constraints = []
        for k in range(self.N):
            constraint = self.prog.AddConstraint(self.eq7c,
                    lb=[0]*3, ub=[0]*3, vars=np.concatenate([q[k], v[k], h[k]]))
            constraint.evaluator().set_description(f"Eq(7c)[{k}]")
            self.eq7c_constraints.append(constraint)

    def check_eq7c_constraints(self, q, v, h):
        return check_constraints(self.eq7c_constraints, {
            "q": q,
            "v": v,
            "h": h
        })

    def eq7d(self, q_qprev_waxis_wmag_v_dt):
        q, qprev, w_axis, w_mag, v, dt = np.split(q_qprev_waxis_wmag_v_dt, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_positions() + 3,
            self.plant_float.num_positions() + self.plant_float.num_positions() + 3 + 1,
            self.plant_float.num_positions() + self.plant_float.num_positions() + 3 + 1 + self.plant_float.num_velocities()])
        plant, context = self.getPlantAndContext(q, v)
        qd = plant.MapVelocityToQDot(context, v*dt[0])
        # return q - qprev - qd
        '''
        As advised in
        https://stackoverflow.com/a/63510131/3177701
        '''
        ret_quat = q[0:4] - apply_angular_velocity_to_quaternion(qprev[0:4], w_axis, w_mag, dt[0])
        ret_linear = (q - qprev - qd)[4:]
        ret = np.hstack([ret_quat, ret_linear])
        return ret

    def add_eq7d_constraints(self):
        q = self.q
        w_axis = self.w_axis
        w_mag = self.w_mag
        v = self.v
        dt = self.dt
        self.eq7d_constraints = []
        for k in range(1, self.N):
            # dt[k] must be converted to an array
            constraint = self.prog.AddConstraint(self.eq7d,
                    lb=[0.0]*self.plant_float.num_positions(),
                    ub=[0.0]*self.plant_float.num_positions(),
                    vars=np.concatenate([q[k], q[k-1], w_axis[k], w_mag[k], v[k], dt[k]]))
            constraint.evaluator().set_description(f"Eq(7d)[{k}]")
            self.eq7d_constraints.append(constraint)

            # Deprecated
            # '''
            # Constrain rotation
            # Taken from Practical Methods for Optimal Control and Estimation by ...
            # Section 6.8 Reorientation of an Asymmetric Rigid Body
            # '''
            # q1 = q[k,0]
            # q2 = q[k,1]
            # q3 = q[k,2]
            # q4 = q[k,3]
            # w1 = v[k,0]*dt[k]
            # w2 = v[k,1]*dt[k]
            # w3 = v[k,2]*dt[k]
            # # Not sure why reshape is necessary
            # self.prog.AddConstraint(eq(q[k,0] - q[k-1,0], 0.5*(w1*q4 - w2*q3 + w3*q2)).reshape((1,)))
            # self.prog.AddConstraint(eq(q[k,1] - q[k-1,1], 0.5*(w1*q3 + w2*q4 - w3*q1)).reshape((1,)))
            # self.prog.AddConstraint(eq(q[k,2] - q[k-1,2], 0.5*(-w1*q2 + w2*q1 + w3*q4)).reshape((1,)))
            # self.prog.AddConstraint(eq(q[k,3] - q[k-1,3], 0.5*(-w1*q1 - w2*q2 - w3*q3)).reshape((1,)))
            # ''' Constrain other positions '''
            # self.prog.AddConstraint(eq(q[k, 4:] - q[k-1, 4:], v[k, 3:]*dt[k]))

    def check_eq7d_constraints(self, q, w_axis, w_mag, v, dt):
        return check_constraints(self.eq7d_constraints, {
            "q": q,
            "w_axis": w_axis,
            "w_mag": w_mag,
            "v": v,
            "dt": dt
        })

    def add_eq7e_constraints(self):
        h = self.h
        hd = self.hd
        dt = self.dt
        self.eq7e_constraints = []
        for k in range(1, self.N):
            constraint = self.prog.AddConstraint(eq(h[k] - h[k-1], hd[k]*dt[k]))
            constraint.evaluator().set_description(f"Eq(7e)[{k}]")
            self.eq7e_constraints.append(constraint)

    def check_eq7e_constraints(self, h, hd, dt):
        return check_constraints(self.eq7e_constraints, {
            "h": h,
            "hd": hd,
            "dt": dt
        })

    def add_eq7f_constraints(self):
        r = self.r
        rd = self.rd
        dt = self.dt
        self.eq7f_constraints = []
        for k in range(1, self.N):
            constraint = self.prog.AddConstraint(
                    eq(r[k] - r[k-1], (rd[k] + rd[k-1])/2*dt[k]))
            constraint.evaluator().set_description(f"Eq(7f)[{k}]")
            self.eq7f_constraints.append(constraint)

    def check_eq7f_constraints(self, r, rd, dt):
        return check_constraints(self.eq7f_constraints, {
            "r": r,
            "rd": rd,
            "dt": dt
        })

    def add_eq7g_constraints(self):
        rd = self.rd
        rdd = self.rdd
        dt = self.dt
        self.eq7g_constraints = []
        for k in range(1, self.N):
            constraint = self.prog.AddConstraint(eq(rd[k] - rd[k-1], rdd[k]*dt[k]))
            constraint.evaluator().set_description(f"Eq(7g)[{k}]")
            self.eq7g_constraints.append(constraint)

    def check_eq7g_constraints(self, rd, rdd, dt):
        return check_constraints(self.eq7g_constraints, {
            "rd": rd,
            "rdd": rdd,
            "dt": dt
        })

    def eq7h(self, q_v_r):
        q, v, r = np.split(q_v_r, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        return  self.calc_r(q, v) - r

    def add_eq7h_constraints(self):
        q = self.q
        v = self.v
        r = self.r
        self.eq7h_constraints = []
        for k in range(self.N):
            # COM position has dimension 3
            constraint = self.prog.AddConstraint(self.eq7h,
                    lb=[0]*3, ub=[0]*3, vars=np.concatenate([q[k], v[k], r[k]]))
            constraint.evaluator().set_description(f"Eq(7h)[{k}]")
            self.eq7h_constraints.append(constraint)

    def check_eq7h_constraints(self, q, v, r):
        return check_constraints(self.eq7h_constraints, {
            "q": q,
            "v": v,
            "r": r
        })

    def eq7i(self, q_v_ck):
        q, v, ck = np.split(q_v_ck, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        cj = self.reshape_3d_contact(ck)
        # print(f"q = {q}\nv={v}\nck={ck}")
        contact_positions = self.get_contact_positions(q, v).T
        return (contact_positions - cj).flatten()

    def add_eq7i_constraints(self):
        q = self.q
        v = self.v
        c = self.c
        self.eq7i_constraints = []
        for k in range(self.N):
            # np.concatenate cannot work q, cj since they have different dimensions
            constraint = self.prog.AddConstraint(self.eq7i,
                    lb=np.zeros(c[k].shape).flatten(), ub=np.zeros(c[k].shape).flatten(),
                    vars=np.concatenate([q[k], v[k], c[k]]))
            constraint.evaluator().set_description(f"Eq(7i)[{k}]")
            self.eq7i_constraints.append(constraint)

    def check_eq7i_constraints(self, q, v, c):
        return check_constraints(self.eq7i_constraints, {
            "q": q,
            "v": v,
            "c": c
        })

    def add_eq7j_constraints(self):
        c = self.c
        self.eq7j_constraints = []
        for k in range(self.N):
            constraint = self.prog.AddBoundingBoxConstraint(
                    [-10, -10, -MAX_GROUND_PENETRATION]*self.num_contacts,
                    [10, 10, 10]*self.num_contacts,
                    c[k])
            constraint.evaluator().set_description(f"Eq(7j)[{k}]")
            self.eq7j_constraints.append(constraint)

    def check_eq7j_constraints(self, c):
        return check_constraints(self.eq7j_constraints, {
            "c": c
        })

    def add_eq7k_admissable_posture_constraints(self):
        q = self.q
        self.eq7k_admissable_posture_constraints = []
        for k in range(self.N):
            constraint = self.prog.AddBoundingBoxConstraint(
                    self.sorted_joint_position_lower_limits,
                    self.sorted_joint_position_upper_limits,
                    q[k, Atlas.FLOATING_BASE_QUAT_DOF:])
            constraint.evaluator().set_description(f"Eq(7k)[{k}] joint position")
            self.eq7k_admissable_posture_constraints.append(constraint)

    def check_eq7k_admissable_posture_constraints(self, q):
        return check_constraints(self.eq7k_admissable_posture_constraints, {
            "q": q
        })

    def add_eq7k_joint_velocity_constraints(self):
        v = self.v
        self.eq7k_joint_velocity_constraints = []
        for k in range(self.N):
            constraint = self.prog.AddBoundingBoxConstraint(
                    -self.sorted_joint_velocity_limits,
                    self.sorted_joint_velocity_limits,
                    v[k, Atlas.FLOATING_BASE_DOF:])
            constraint.evaluator().set_description(f"Eq(7k)[{k}] joint velocity")
            self.eq7k_joint_velocity_constraints.append(constraint)

    def check_eq7k_joint_velocity_constraints(self, v):
        return check_constraints(self.eq7k_joint_velocity_constraints, {
            "v": v
        })

    def add_eq7k_friction_cone_constraints(self):
        F = self.F
        beta = self.beta
        self.eq7k_friction_cone_constraints = []
        for k in range(self.N):
            Fj = self.reshape_3d_contact(F[k])
            beta_k = np.reshape(beta[k], (self.num_contacts, self.N_d))
            for i in range(self.num_contacts):
                beta_v = beta_k[i].dot(self.friction_cone_components[:,i,:])
                constraint = self.prog.AddLinearConstraint(eq(Fj[i], beta_v))
                constraint.evaluator().set_description(f"Eq(7k)[{k}] friction cone constraint[{i}]")
                self.eq7k_friction_cone_constraints.append(constraint)

    def check_eq7k_friction_cone_constraints(self, F, beta):
        return check_constraints(self.eq7k_friction_cone_constraints, {
            "F": F,
            "beta": beta
        })

    def add_eq7k_beta_positive_constraints(self):
        beta = self.beta
        self.eq7k_beta_positive_constraints = []
        for k in range(self.N):
            for b in beta[k]:
                constraint = self.prog.AddLinearConstraint(b >= 0.0)
                constraint.evaluator().set_description(f"Eq(7k)[{k}] beta >= 0 constraint")
                self.eq7k_beta_positive_constraints.append(constraint)

    def check_eq7k_beta_positive_constraints(self, beta):
        return check_constraints(self.eq7k_beta_positive_constraints, {
            "beta": beta
        })

    def add_eq7k_torque_constraints(self):
        tau = self.tau
        beta = self.beta
        self.eq7k_torque_constraints = []
        for k in range(self.N):
            ''' Constrain torques - assume torque linear to friction cone'''
            beta_k = np.reshape(beta[k], (self.num_contacts, self.N_d))
            friction_torque_constraints = []
            for i in range(self.num_contacts):
                max_torque = friction_torque_coefficient * np.sum(beta_k[i])
                upper_constraint = self.prog.AddLinearConstraint(le(tau[k][i], np.array([max_torque])))
                upper_constraint.evaluator().set_description(f"Eq(7k)[{k}] friction torque upper limit")
                lower_constraint = self.prog.AddLinearConstraint(ge(tau[k][i], np.array([-max_torque])))
                lower_constraint.evaluator().set_description(f"Eq(7k)[{k}] friction torque lower limit")
                friction_torque_constraints.append([upper_constraint, lower_constraint])
            self.eq7k_torque_constraints.append(friction_torque_constraints)

    def check_eq7k_torque_constraints(self, tau, beta):
        return check_constraints(self.eq7k_torque_constraints, {
            "tau": tau,
            "beta": beta
        })

    def add_eq8a_constraints(self):
        def eq8a_lhs(F, c, slack):
            Fj = self.reshape_3d_contact(F)
            cj = self.reshape_3d_contact(c)
            return Fj[:,2].dot(cj[:,2]) - slack # Constraint functions must output vectors

        F = self.F
        c = self.c
        slack = self.slack
        self.eq8a_constraints = []
        for k in range(self.N):
            constraint = self.add_constraint(eq8a_lhs, F[k], c[k], slack[k])
            constraint.evaluator().set_description(f"Eq(8a)[{k}]")
            self.eq8a_constraints.append(constraint)

    def check_eq8a_constraints(self, F, c, slack):
        return check_constraints(self.eq8a_constraints, {
            "F": F,
            "c": c,
            "slack": slack
        })

    def add_eq8b_constraints(self):
        tau = self.tau
        c = self.c
        slack = self.slack
        self.eq8b_constraints = []
        for k in range(self.N):
            constraint = self.add_constraint(
                    lambda tau, c, slack : (tau**2).T.dot(self.reshape_3d_contact(c)[:,2]) - slack,
                    tau[k], c[k], slack[k])
            constraint.evaluator().set_description(f"Eq(8b)[{k}]")
            self.eq8b_constraints.append(constraint)

    def check_eq8b_constraints(self, tau, c, slack):
        return check_constraints(self.eq8b_constraints, {
            "tau": tau,
            "c": c,
            "slack": slack
        })

    def add_eq8c_contact_force_constraints(self):
        F = self.F
        self.eq8c_contact_force_constraints = []
        for k in range(self.N):
            Fj = self.reshape_3d_contact(F[k])
            constraint = self.prog.AddLinearConstraint(ge(Fj[:,2], 0.0))
            constraint.evaluator().set_description(f"Eq(8c)[{k}] contact force greater than zero")
            self.eq8c_contact_force_constraints.append(constraint)

    def check_eq8c_contact_force_constraints(self, F):
        return check_constraints(self.eq8c_contact_force_constraints, {
            "F": F
        })

    def add_eq8c_contact_distance_constraints(self):
        c = self.c
        self.eq8c_contact_distance_constraints = []
        for k in range(self.N):
            # TODO: Why can't this be converted to a linear / boundingbox constraint?

            # This doesn't work due to mixing of autodiff plant and Variable
            # constraint = self.prog.AddConstraint(
                    # ge(self.get_contact_positions_z(q[k], v[k]), -MAX_GROUND_PENETRATION))
            constraint = self.add_constraint(
                    lambda c : self.reshape_3d_contact(c)[:,2],
                    c[k],
                    lb=[-MAX_GROUND_PENETRATION] * self.num_contacts,
                    ub=[float('inf')] * self.num_contacts)
            constraint.evaluator().set_description(f"Eq(8c)[{k}] z position greater than zero")
            self.eq8c_contact_distance_constraints.append(constraint)

    def check_eq8c_contact_distance_constraints(self, c):
        return check_constraints(self.eq8c_contact_distance_constraints, {
            "c": c
        })

    def add_eq9a_constraints(self):
        ''' Assume flat ground for now... '''
        def eq9a_lhs(F, c, c_prev, i, slack):
            Fj = self.reshape_3d_contact(F)
            cj = self.reshape_3d_contact(c)
            cj_prev = self.reshape_3d_contact(c_prev)
            return Fj[i,2] * (cj[i] - cj_prev[i]).dot(np.array([1.0, 0.0, 0.0])) - slack

        F = self.F
        c = self.c
        slack = self.slack
        self.eq9a_constraints = []
        for k in range(1, self.N):
            contact_constraints = []
            for i in range(self.num_contacts):
                '''
                i=i is used to capture the outer scope i variable
                https://stackoverflow.com/a/2295372/3177701
                '''

                # This doesn't work because MathematicalProgram.AddConstraint doesn't like vars having non-Variable types
                # constraint = self.add_constraint(eq9a_lhs, F[k], c[k], c[k-1], i, slack)

                constraint = self.add_constraint(
                        lambda F, c, cprev, slack, i=i : eq9a_lhs(F, c, cprev, i, slack),
                        F[k], c[k], c[k-1], slack[k])
                constraint.evaluator().set_description("Eq(9a)[{k}][{i}]")
                contact_constraints.append(constraint)
            self.eq9a_constraints.append(contact_constraints)

    def check_eq9a_constraints(self, F, c, slack):
        return check_constraints(self.eq9a_constraints, {
            "F": F,
            "c": c,
            "slack": slack
        })

    def add_eq9b_constraints(self):
        def eq9b_lhs(F, c, c_prev, i, slack):
            Fj = self.reshape_3d_contact(F)
            cj = self.reshape_3d_contact(c)
            cj_prev = self.reshape_3d_contact(c_prev)
            return Fj[i,2] * (cj[i] - cj_prev[i]).dot(np.array([0.0, 1.0, 0.0])) - slack

        F = self.F
        c = self.c
        slack = self.slack
        self.eq9b_constraints = []
        for k in range(1, self.N):
            contact_constraints = []
            for i in range(self.num_contacts):
                '''
                i=i is used to capture the outer scope i variable
                https://stackoverflow.com/a/2295372/3177701
                '''
                constraint = self.add_constraint(
                        lambda F, c, cprev, slack, i=i : eq9b_lhs(F, c, cprev, i, slack),
                        F[k], c[k], c[k-1], slack[k])
                constraint.evaluator().set_description("Eq(9b)[{k}][{i}]")
                contact_constraints.append(constraint)
            self.eq9b_constraints.append(contact_constraints)

    def check_eq9b_constraints(self, F, c, slack):
        return check_constraints(self.eq9b_constraints, {
            "F": F,
            "c": c,
            "slack": slack
        })

    def add_slack_constraints(self):
        self.prog.AddConstraint(ge(self.slack, 0))

    def add_contact_sequence_constraint(self):
        right_foot_start_idx = int(self.num_contacts/2)
        c = self.c
        self.contact_sequence_constraints = []
        for k in range(self.N):
            cj = self.reshape_3d_contact(c[k])
            if k < int(self.N/5):
                # Double stance
                double_stance_constraint = self.prog.AddLinearConstraint(eq(cj[:, 2], 0.0))
                double_stance_constraint.evaluator().set_description(f"double stance 1")
                self.contact_sequence_constraints.append(double_stance_constraint)
            elif k < int(2*self.N/5):
                # Left swing, right stance
                left_swing_constraint = self.prog.AddLinearConstraint(ge(cj[:right_foot_start_idx, 2], 0.0))
                left_swing_constraint.evaluator().set_description(f"left swing")
                self.contact_sequence_constraints.append(left_swing_constraint)
                right_stance_constraint = self.prog.AddLinearConstraint(eq(cj[right_foot_start_idx:, 2], 0.0))
                right_stance_constraint.evaluator().set_description(f"right stance")
                self.contact_sequence_constraints.append(right_stance_constraint)
            elif k < int(3*self.N/5):
                # Double stance
                double_stance_constraint = self.prog.AddLinearConstraint(eq(cj[:,2], 0.0))
                double_stance_constraint.evaluator().set_description(f"double stance 2")
                self.contact_sequence_constraints.append(double_stance_constraint)
            elif k < int(4*self.N/5):
                # Left stance right swing
                left_swing_constraint = self.prog.AddLinearConstraint(eq(cj[:right_foot_start_idx, 2], 0.0))
                left_swing_constraint.evaluator().set_description(f"left swing")
                self.contact_sequence_constraints.append(left_swing_constraint)
                right_stance_constraint = self.prog.AddLinearConstraint(ge(cj[right_foot_start_idx:, 2], 0.0))
                right_stance_constraint.evaluator().set_description(f"right stance")
                self.contact_sequence_constraints.append(right_stance_constraint)
            else:
                # Double stance
                double_stance_constraint = self.prog.AddLinearConstraint(eq(cj[:,2], 0.0))
                double_stance_constraint.evaluator().set_description(f"double stance 3")
                self.contact_sequence_constraints.append(double_stance_constraint)

    def add_initial_pose_constraints(self, q_init):
        self.q_init = q_init
        q = self.q
        self.initial_pose_constraints = []
        constraint = self.prog.AddLinearConstraint(eq(q[0], self.q_init))
        constraint.evaluator().set_description("initial pose")
        self.initial_pose_constraints.append(constraint)

    def check_initial_pose_constraints(self, q):
        return check_constraints(self.initial_pose_constraints, {
            "q": q,
        })

    def add_initial_velocity_constraints(self):
        v = self.v
        self.initial_velocity_constraints = []
        constraint = self.prog.AddLinearConstraint(eq(v[0], 0.0))
        constraint.evaluator().set_description("initial velocity")
        self.initial_velocity_constraints.append(constraint)

    def check_initial_velocity_constraints(self, v):
        return check_constraints(self.initial_velocity_constraints, {
            "v": v
        })

    def add_final_pose_constraints(self, q_final, pelvis_only):
        self.q_final = q_final
        q = self.q
        self.final_pose_constraints = []
        if pelvis_only:
            constraint = self.prog.AddLinearConstraint(eq(q[-1, 0:7], self.q_final[0:7]))
            constraint.evaluator().set_description("final pose")
        else:
            constraint = self.prog.AddLinearConstraint(eq(q[-1], self.q_final))
            constraint.evaluator().set_description("final pose")
        self.final_pose_constraints.append(constraint)

    def check_final_pose_constraints(self, q):
        return check_constraints(self.final_pose_constraints, {
            "q": q
        })

    def add_final_velocity_constraints(self):
        v = self.v
        self.final_velocity_constraints = []
        constraint = self.prog.AddLinearConstraint(eq(v[-1], 0.0))
        constraint.evaluator().set_description("final velocity")
        self.final_velocity_constraints.append(constraint)

    def check_final_velocity_constraints(self, v):
        return check_constraints(self.final_velocity_constraints, {
            "v": v
        })

    def add_final_COM_velocity_constraints(self):
        rd = self.rd
        self.final_COM_velocity_constraints = []
        constraint = self.prog.AddLinearConstraint(eq(rd[-1], 0.0))
        constraint.evaluator().set_description("final COM velocity")
        self.final_COM_velocity_constraints.append(constraint)

    def check_final_COM_velocity_constraints(self, rd):
        return check_constraints(self.final_COM_velocity_constraints, {
            "rd": rd
        })

    def add_final_COM_acceleration_constraints(self):
        rdd = self.rdd
        self.final_COM_acceleration_constraints = []
        constraint = self.prog.AddLinearConstraint(eq(rdd[-1], 0.0))
        constraint.evaluator().set_description("final COM acceleration")
        self.final_COM_acceleration_constraints.append(constraint)

    def check_final_COM_acceleration_constraints(self, rdd):
        return check_constraints(self.final_COM_acceleration_constraints, {
            "rdd": rdd
        })

    def add_max_time_constraints(self, max_time):
        self.T = max_time
        dt = self.dt
        self.max_time_constraints = []
        constraint = self.prog.AddLinearConstraint(np.sum(dt) == self.T)
        constraint.evaluator().set_description("max time")
        self.max_time_constraints.append(constraint)

    def check_max_time_constraints(self, dt):
        return check_constraints(self.max_time_constraints, {
            "dt": dt
        })

    def add_timestep_constraints(self):
        dt = self.dt
        self.timestep_constraints = []

        # Note that the first time step is only used in the initial cost calculation
        # and not in the backwards Euler
        constraint = self.prog.AddLinearConstraint(eq(dt[0], 0.0))
        constraint.evaluator().set_description("first timestep")
        self.timestep_constraints.append(constraint)

        # Values taken from
        # https://colab.research.google.com/github/RussTedrake/underactuated/blob/master/exercises/simple_legs/compass_gait_limit_cycle/compass_gait_limit_cycle.ipynb
        constraint = self.prog.AddLinearConstraint(ge(dt[1:], [[MIN_TIMESTEP]]*(self.N-1)))
        constraint.evaluator().set_description("min timestep")
        self.timestep_constraints.append(constraint)
        constraint = self.prog.AddLinearConstraint(le(dt, [[MAX_TIMESTEP]]*self.N))
        constraint.evaluator().set_description("max timestep")
        self.timestep_constraints.append(constraint)

    def check_timestep_constraints(self, dt):
        return check_constraints(self.timestep_constraints, {
            "dt": dt
        })

    def add_joint_acceleration_constraints(self):
        v = self.v
        dt = self.dt
        self.joint_acceleration_constraints = []
        ''' Constrain max joint acceleration '''
        for k in range(1, self.N):
            constraint = self.prog.AddLinearConstraint(ge((v[k] - v[k-1])[6:], -MAX_JOINT_ACCELERATION*dt[k]))
            constraint.evaluator().set_description(f"min joint acceleration[{k}]")
            self.joint_acceleration_constraints.append(constraint)
            constraint = self.prog.AddLinearConstraint(le((v[k] - v[k-1])[6:], MAX_JOINT_ACCELERATION*dt[k]))
            constraint.evaluator().set_description(f"max joint acceleration[{k}]")
            self.joint_acceleration_constraints.append(constraint)

    def check_joint_acceleration_constraints(self, v, dt):
        return check_constraints(self.joint_acceleration_constraints, {
            "v": v,
            "dt": dt
        })

    def add_unit_quaternion_constraints(self):
        q = self.q
        self.unit_quaternion_constraints = []
        for k in range(self.N):
            AddUnitQuaternionConstraintOnPlant(self.plant_float, q[k], self.prog)
            # constraint.evaluator().set_description(f"unit quaternion constraint[{k}]")
            # self.unit_quaternion_constraints.append(constraint)

    def check_unit_quaternion_constraints(self, q):
        return check_constraints(self.unit_quaternion_constraints, {
            "q": q
        })

    def add_angular_velocity_constraints(self):
        v = self.v
        w_axis = self.w_axis
        w_mag = self.w_mag
        self.angular_velocity_constraints = []
        for k in range(self.N):
            constraint = self.prog.AddConstraint(eq(v[k][0:3], w_axis[k] * w_mag[k]))
            constraint.evaluator().set_description(f"angular velocity constraint[{k}]")
            self.angular_velocity_constraints.append(constraint)

    def check_angular_velocity_constraints(self, v, w_axis, w_mag):
        return check_constraints(self.angular_velocity_constraints, {
            "v": v,
            "w_axis": w_axis,
            "w_mag": w_mag
        })

    def add_unit_axis_constraint(self):
        w_axis = self.w_axis
        self.unit_axis_constraints = []
        for k in range(self.N):
            constraint = self.prog.AddConstraint(lambda x: [x @ x], [1], [1], w_axis[k])
            constraint.evaluator().set_description(f"unit axis constraint[{k}]")
            self.unit_axis_constraints.append(constraint)

    def check_unit_axis_constraint(self, w_axis):
        return check_constraints(self.unit_axis_constraints, {
            "w_axis": w_axis
        })

    def check_all_constraints(self, q, w_axis, w_mag, v, dt, r, rd, rdd, c, F, tau, h, hd, beta):
        return (self.check_eq7a_constraints(F, rdd)
                and self.check_eq7b_constraints(F, c, tau, hd, r)
                and self.check_eq7c_constraints(q, v, h)
                and self.check_eq7d_constraints(q, w_axis, w_mag, v, dt)
                and self.check_eq7e_constraints(h, hd, dt)
                and self.check_eq7f_constraints(r, rd, dt)
                and self.check_eq7g_constraints(rd, rdd, dt)
                and self.check_eq7h_constraints(q, v, r)
                and self.check_eq7i_constraints(q, v, c)
                and self.check_eq7j_constraints(c)
                and self.check_eq7k_admissable_posture_constraints(q)
                and self.check_eq7k_joint_velocity_constraints(v)
                and self.check_eq7k_friction_cone_constraints(F, beta)
                and self.check_eq7k_beta_positive_constraints(beta)
                and self.check_eq7k_torque_constraints(tau, beta)
                and self.check_eq8a_constraints(q, v, F)
                and self.check_eq8b_constraints(q, v, tau)
                and self.check_eq8c_contact_force_constraints(F)
                and self.check_eq8c_contact_distance_constraint(q, v)
                and self.check_eq9a_constraints(F, c)
                and self.check_eq9b_constraints(F, c)
                and self.check_initial_pose_constraints(q)
                and self.check_initial_velocity_constraints(v)
                and self.check_final_pose_constraints(q)
                and self.check_final_velocity_constraints(v)
                and self.check_final_COM_velocity_constraints(rd)
                and self.check_final_COM_acceleration_constraints(rdd)
                and self.check_max_time_constraints(dt)
                and self.check_timestep_constraints(dt)
                and self.check_joint_acceleration_constraints(v, dt)
                and self.check_unit_quaternion_constraints(q)
                and self.check_angular_velocity_constraints(v, w_axis, w_mag)
                and self.check_unit_axis_constraint(w_axis))

    def add_eq10_cost(self):
        q = self.q
        v = self.v
        dt = self.dt
        rdd = self.rdd
        for k in range(self.N):
            Q_v = 0.5 * np.identity(self.plant_float.num_velocities())
            self.prog.AddCost(self.pose_error_cost, vars=np.concatenate([q[k], v[k], dt[k]])) # np.concatenate requires items to have compatible shape
            self.prog.AddCost((dt[k]*(
                    + v[k].dot(Q_v).dot(v[k])
                    + rdd[k].dot(rdd[k])))[0])

    def add_slack_cost(self):
        self.prog.AddCost(1e3*(self.slack.T@self.slack)[0,0])

    def create_minimal_program(self, num_knot_points, max_time):
        assert(max_time / num_knot_points > MIN_TIMESTEP)
        assert(max_time / num_knot_points < MAX_TIMESTEP)
        self.N = num_knot_points

        self.prog = MathematicalProgram()
        self.q = self.prog.NewContinuousVariables(rows=self.N, cols=self.plant_float.num_positions(), name="q")
        # Special variables for handling floating base angular velocity
        self.w_axis = self.prog.NewContinuousVariables(rows=self.N, cols=3, name="w_axis")
        self.w_mag = self.prog.NewContinuousVariables(rows=self.N, cols=1, name="w_mag")
        self.v = self.prog.NewContinuousVariables(rows=self.N, cols=self.plant_float.num_velocities(), name="v")
        self.dt = self.prog.NewContinuousVariables(rows=self.N, cols=1, name="dt")
        self.r = self.prog.NewContinuousVariables(rows=self.N, cols=3, name="r")
        self.rd = self.prog.NewContinuousVariables(rows=self.N, cols=3, name="rd")
        self.rdd = self.prog.NewContinuousVariables(rows=self.N, cols=3, name="rdd")
        # The cols are ordered as
        # [contact1_x, contact1_y, contact1_z, contact2_x, contact2_y, contact2_z, ...]
        self.c = self.prog.NewContinuousVariables(rows=self.N, cols=self.contact_dim, name="c")
        self.F = self.prog.NewContinuousVariables(rows=self.N, cols=self.contact_dim, name="F")
        self.tau = self.prog.NewContinuousVariables(rows=self.N, cols=self.num_contacts, name="tau") # We assume only z torque exists
        self.h = self.prog.NewContinuousVariables(rows=self.N, cols=3, name="h")
        self.hd = self.prog.NewContinuousVariables(rows=self.N, cols=3, name="hd")

        ''' Additional variables not explicitly stated '''
        # Friction cone scale
        self.beta = self.prog.NewContinuousVariables(rows=self.N, cols=self.num_contacts*self.N_d, name="beta")

        ''' These constraints were not explicitly stated in the paper'''
        self.add_max_time_constraints(max_time)
        self.add_timestep_constraints()

    def add_0th_order_constraints(self, q_init, q_final, pelvis_only):
        ''' These constraints were not explicitly stated in the paper'''
        self.add_initial_pose_constraints(q_init)
        self.add_initial_velocity_constraints()
        self.add_final_pose_constraints(q_final, pelvis_only)
        self.add_final_velocity_constraints()
        self.add_final_COM_velocity_constraints()
        self.add_final_COM_acceleration_constraints()
        self.add_joint_acceleration_constraints()
        self.add_unit_quaternion_constraints()
        self.add_angular_velocity_constraints()
        self.add_unit_axis_constraint()

    def create_full_program(self, q_init, q_final, num_knot_points, max_time, pelvis_only=True):
        self.create_minimal_program(num_knot_points, max_time)
        self.add_eq7a_constraints()
        self.add_eq7b_constraints()
        self.add_eq7c_constraints()
        self.add_eq7d_constraints()
        self.add_eq7e_constraints()
        self.add_eq7f_constraints()
        self.add_eq7g_constraints()
        self.add_eq7h_constraints()
        self.add_eq7i_constraints()
        self.add_eq7j_constraints()
        self.add_eq7k_admissable_posture_constraints()
        self.add_eq7k_joint_velocity_constraints()
        self.add_eq7k_friction_cone_constraints()
        self.add_eq7k_beta_positive_constraints()
        self.add_eq7k_torque_constraints()
        if ENABLE_COMPLEMENTARITY_CONSTRAINTS:
            self.slack = self.prog.NewContinuousVariables(rows=self.N, cols=1, name="slack")
            self.add_eq8a_constraints()
            self.add_eq8b_constraints()
            self.add_eq8c_contact_force_constraints()
            self.add_eq8c_contact_distance_constraints()
            self.add_eq9a_constraints()
            self.add_eq9b_constraints()
            self.add_slack_constraints()
            self.add_slack_cost()
        else:
            self.add_contact_sequence_constraint()

        self.add_eq10_cost()

        self.add_0th_order_constraints(q_init, q_final, pelvis_only)

        '''
        Constrain unbounded variables to improve IPOPT performance
        because IPOPT is an interior point method which works poorly for unbounded variables
        '''
        # (self.prog.AddLinearConstraint(le(F.flatten(), np.ones(F.shape).flatten()*1e3))
                # .evaluator().set_description("max F"))
        # (self.prog.AddBoundingBoxConstraint(-1e3, 1e3, tau)
                # .evaluator().set_description("bound tau"))
        # (self.prog.AddLinearConstraint(le(beta.flatten(), np.ones(beta.shape).flatten()*1e3))
                # .evaluator().set_description("max beta"))

    def solve(self):
        ''' Guess '''
        initial_guess = np.empty(self.prog.num_vars())
        dt_guess = [0.0] + [self.T/(self.N-1)] * (self.N-1)
        self.prog.SetDecisionVariableValueInVector(self.dt, dt_guess, initial_guess)
        # Guess q to avoid initializing with invalid quaternion
        quat_traj_guess = PiecewiseQuaternionSlerp()
        quat_traj_guess.Append(0, Quaternion(self.q_init[0:4]))
        quat_traj_guess.Append(self.T, Quaternion(self.q_final[0:4]))
        position_traj_guess = PiecewisePolynomial.FirstOrderHold([0.0, self.T], np.vstack([self.q_init[4:], self.q_final[4:]]).T)
        q_guess = np.array([np.hstack([
            Quaternion(quat_traj_guess.value(t)).wxyz(), position_traj_guess.value(t).flatten()])
            for t in np.linspace(0, self.T, self.N)])
        self.prog.SetDecisionVariableValueInVector(self.q, q_guess, initial_guess)

        w_axis_guess = np.array([[0.0, 0.0, 1.0]] * self.N)
        self.prog.SetDecisionVariableValueInVector(self.w_axis, w_axis_guess, initial_guess)

        v_traj_guess = position_traj_guess.MakeDerivative()
        w_traj_guess = quat_traj_guess.MakeDerivative()
        v_guess = np.array([
            np.hstack([w_traj_guess.value(t).flatten(), v_traj_guess.value(t).flatten()])
            for t in np.linspace(0, self.T, self.N)])
        self.prog.SetDecisionVariableValueInVector(self.v, v_guess, initial_guess)

        c_guess = np.array([
            self.get_contact_positions(q_guess[i], v_guess[i]).T.flatten() for i in range(self.N)])
        for i in range(self.N):
            assert((self.eq7i(np.concatenate([q_guess[i], v_guess[i], c_guess[i]])) == 0.0).all())
        self.prog.SetDecisionVariableValueInVector(self.c, c_guess, initial_guess)

        r_guess = np.array([
            self.calc_r(q_guess[i], v_guess[i]) for i in range(self.N)])
        self.prog.SetDecisionVariableValueInVector(self.r, r_guess, initial_guess)

        h_guess = np.array([
            self.calc_h(q_guess[i], v_guess[i]) for i in range(self.N)])
        self.prog.SetDecisionVariableValueInVector(self.h, h_guess, initial_guess)

        F_guess = np.zeros((self.N, self.contact_dim))
        F_guess[:,2::3] = Atlas.M * Atlas.g / self.num_contacts
        self.prog.SetDecisionVariableValueInVector(self.F, F_guess, initial_guess)

        if ENABLE_COMPLEMENTARITY_CONSTRAINTS:
            slack_guess = [0.1] * self.N
            self.prog.SetDecisionVariableValueInVector(self.slack, slack_guess, initial_guess)

        ''' Solve '''
        solver = SnoptSolver()
        options = SolverOptions()
        # options.SetOption(solver.solver_id(), "max_iter", 50000)
        # This doesn't seem to do anything...
        # options.SetOption(CommonSolverOption.kPrintToConsole, True)
        start_solve_time = time.time()
        print(f"Start solving...")
        result = solver.Solve(self.prog, initial_guess, options) # Currently takes around 30 mins
        print(f"Success: {result.is_success()}  Solve time: {time.time() - start_solve_time}s  Cost: {result.get_optimal_cost()}")
        self.q_sol = result.GetSolution(self.q)
        self.v_sol = result.GetSolution(self.v)
        self.dt_sol = result.GetSolution(self.dt)
        self.r_sol = result.GetSolution(self.r)
        self.rd_sol = result.GetSolution(self.rd)
        self.rdd_sol = result.GetSolution(self.rdd)
        self.c_sol = result.GetSolution(self.c)
        self.F_sol = result.GetSolution(self.F)
        self.tau_sol = result.GetSolution(self.tau)
        self.h_sol = result.GetSolution(self.h)
        self.hd_sol = result.GetSolution(self.hd)
        self.beta_sol = result.GetSolution(self.beta)
        if ENABLE_COMPLEMENTARITY_CONSTRAINTS:
            self.slack_sol = result.GetSolution(self.slack)
        if not result.is_success():
            print(result.GetInfeasibleConstraintNames(self.prog))
            pdb.set_trace()

        return (self.r_sol, self.rd_sol, self.rdd_sol,
                self.q_sol, self.v_sol, self.dt_sol)

def main():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(mbp_time_step))
    load_atlas(plant, add_ground=True)
    plant_context = plant.CreateDefaultContext()

    upright_context = plant.CreateDefaultContext()
    set_atlas_initial_pose(plant, upright_context)
    q_nom = plant.GetPositions(upright_context)
    q_init = q_nom.copy()
    q_init[6] = 0.93846 # Avoid initializing with ground penetration
    q_final = q_init.copy()
    q_final[4] = 0.0 # x position of pelvis
    q_final[6] = 0.93845 # z position of pelvis (to make sure final pose touches ground)

    num_knot_points = 50
    max_time = 0.5

    export_filename = f"sample(final_x_{q_final[4]})(num_knot_points_{num_knot_points})(max_time_{max_time})"

    planner = HumanoidPlanner(plant, Atlas.CONTACTS_PER_FRAME, q_nom)
    if not PLAYBACK_ONLY:
        print(f"Starting pos: {q_init}\nFinal pos: {q_final}")
        planner.create_full_program(q_init, q_final, num_knot_points, max_time, pelvis_only=True)
        r_traj, rd_traj, rdd_traj, q_traj, v_traj, dt_traj = planner.solve()

        with open(export_filename, 'wb') as f:
            pickle.dump([r_traj, rd_traj, rdd_traj, q_traj, v_traj, dt_traj], f)

    with open(export_filename, 'rb') as f:
        r_traj, rd_traj, rdd_traj, q_traj, v_traj, dt_traj = pickle.load(f)

    controller = builder.AddSystem(HumanoidController(plant, Atlas.CONTACTS_PER_FRAME, is_wbc=True))
    controller.set_name("HumanoidController")

    ''' Connect atlas plant to controller '''
    builder.Connect(plant.get_state_output_port(), controller.GetInputPort("q_v"))
    builder.Connect(controller.GetOutputPort("tau"), plant.get_actuation_input_port())

    quaternion_poly, position_poly = create_q_interpolation(plant, plant_context, q_traj, v_traj, dt_traj)

    quaternion_source = builder.AddSystem(TrajectorySource(quaternion_poly))
    position_source = builder.AddSystem(TrajectorySource(position_poly))
    q_multiplexer = builder.AddSystem(Multiplexer([4, plant.num_positions() - 4]))
    builder.Connect(quaternion_source.get_output_port(), q_multiplexer.get_input_port(0))
    builder.Connect(position_source.get_output_port(), q_multiplexer.get_input_port(1))
    builder.Connect(q_multiplexer.get_output_port(0), controller.GetInputPort("q_des"))

    positiond_source = builder.AddSystem(TrajectorySource(position_poly.MakeDerivative()))
    quaterniond_source = builder.AddSystem(TrajectorySource(quaternion_poly.MakeDerivative()))
    v_multiplexer = builder.AddSystem(Multiplexer([3, plant.num_velocities() - 3]))
    builder.Connect(quaterniond_source.get_output_port(), v_multiplexer.get_input_port(0))
    builder.Connect(positiond_source.get_output_port(), v_multiplexer.get_input_port(1))
    builder.Connect(v_multiplexer.get_output_port(0), controller.GetInputPort("v_des"))

    positiondd_source = builder.AddSystem(TrajectorySource(position_poly.MakeDerivative().MakeDerivative()))
    quaterniondd_source = builder.AddSystem(TrajectorySource(quaternion_poly.MakeDerivative().MakeDerivative()))
    vd_multiplexer = builder.AddSystem(Multiplexer([3, plant.num_velocities() - 3]))
    builder.Connect(quaterniondd_source.get_output_port(), vd_multiplexer.get_input_port(0))
    builder.Connect(positiondd_source.get_output_port(), vd_multiplexer.get_input_port(1))
    builder.Connect(vd_multiplexer.get_output_port(0), controller.GetInputPort("vd_des"))

    r_poly, rd_poly, rdd_poly = create_r_interpolation(r_traj, rd_traj, rdd_traj, dt_traj)
    r_source = builder.AddSystem(TrajectorySource(r_poly))
    rd_source = builder.AddSystem(TrajectorySource(rd_poly))
    rdd_source = builder.AddSystem(TrajectorySource(rdd_poly))
    builder.Connect(r_source.get_output_port(0), controller.GetInputPort("r_des"))
    builder.Connect(rd_source.get_output_port(0), controller.GetInputPort("rd_des"))
    builder.Connect(rdd_source.get_output_port(0), controller.GetInputPort("rdd_des"))

    ConnectContactResultsToDrakeVisualizer(builder=builder, plant=plant)
    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
    diagram = builder.Build()

    frame_idx = 0
    t = 0.0
    while True:
        print(f"Frame: {frame_idx}, t: {t}, dt: {dt_traj[frame_idx]}")
        diagram_context = diagram.CreateDefaultContext()
        plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
        plant.SetPositions(plant_context, q_traj[frame_idx])

        simulator = Simulator(diagram, diagram_context)
        simulator.set_target_realtime_rate(0.0)
        simulator.AdvanceTo(0)
        pdb.set_trace()

        frame_idx = (frame_idx + 1) % q_traj.shape[0]
        t = t + dt_traj[frame_idx]

if __name__ == "__main__":
    main()
