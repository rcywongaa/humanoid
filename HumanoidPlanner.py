#!/usr/bin/python3

'''
This implements the paper
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics
by Hongkai Dai, Andrés Valenzuela and Russ Tedrake
'''

from Atlas import load_atlas, set_atlas_initial_pose
from Atlas import getSortedJointLimits, getActuatorIndex, getActuatorIndices, getJointValues
from Atlas import Atlas
from pydrake.all import Quaternion
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

mbp_time_step = 1.0e-3
N_d = 4 # friction cone approximated as a i-pyramid
N_f = 3 # contact force dimension
mu = 1.0 # Coefficient of friction
epsilon = 1e-9
PLAYBACK_ONLY = False
ENABLE_COMPLEMENTARITY_CONSTRAINTS = True
MAX_GROUND_PENETRATION = 1e-2
MAX_JOINT_ACCELERATION = 20.0
'''
Slack for the complementary constraints
Same value used in drake/multibody/optimization/static_equilibrium_problem.cc
'''
# slack = 1e-3
slack = 1e-2

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

def apply_angular_velocity_to_quaternion(q, w, t):
    # This currently returns a runtime warning of division by zero
    # https://github.com/RobotLocomotion/drake/issues/10451
    norm_w = np.linalg.norm(w)
    if norm_w <= epsilon:
        return q
    a = w / norm_w
    if q.dtype == AutoDiffXd:
        delta_q = Quaternion_[AutoDiffXd](np.hstack([np.cos(norm_w * t/2.0), a*np.sin(norm_w * t/2.0)]).reshape((4,1)))
        return Quaternion_[AutoDiffXd](q/np.linalg.norm(q)).multiply(delta_q).wxyz()
    else:
        delta_q = Quaternion(np.hstack([np.cos(norm_w * t/2.0), a*np.sin(norm_w * t/2.0)]).reshape((4,1)))
        return Quaternion(q/np.linalg.norm(q)).multiply(delta_q).wxyz()

class HumanoidPlanner:
    def __init__(self, plant_float, contacts_per_frame, q_nom):
        self.plant_float = plant_float
        self.context_float = plant_float.CreateDefaultContext()
        self.plant_autodiff = self.plant_float.ToAutoDiffXd()
        self.context_autodiff = self.plant_autodiff.CreateDefaultContext()
        self.q_nom = q_nom

        self.contacts_per_frame = contacts_per_frame
        self.num_contacts = sum([contact_points.shape[1] for contact_points in contacts_per_frame.values()])
        self.contact_dim = 3*self.num_contacts

        n = np.array([
            [0],
            [0],
            [1.0]])
        d = np.array([
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0]])
        # Equivalent to v in HumanoidController.py
        self.friction_cone_components = np.zeros((N_d, self.num_contacts, N_f))
        for i in range(N_d):
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
    Creates an np.array of shape [num_contacts, 3] where first 2 rows are zeros
    since we only care about tau in the z direction
    '''
    def toTauj(self, tau_k):
        return np.hstack([np.zeros((self.num_contacts, 2)), np.reshape(tau_k, (self.num_contacts, 1))])

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

    ''' Assume flat ground for now... '''
    def get_contact_positions_z(self, q, v):
        return self.get_contact_positions(q, v)[2,:]

    # https://stackoverflow.com/questions/63454077/how-to-obtain-centroidal-momentum-matrix/63456202#63456202
    def calc_h(self, q, v):
        plant, context = self.getPlantAndContext(q, v)
        return plant.CalcSpatialMomentumInWorldAboutPoint(context, plant.CalcCenterOfMassPosition(context)).rotational()

    def calc_r(self, q, v):
        plant, context = self.getPlantAndContext(q, v)
        return plant.CalcCenterOfMassPosition(context)

    def eq7c(self, q_v_h):
        q, v, h = np.split(q_v_h, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        return self.calc_h(q, v) - h

    def eq7d(self, q_qprev_v_dt):
        q, qprev, v, dt = np.split(q_qprev_v_dt, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_positions() + self.plant_float.num_velocities()])
        plant, context = self.getPlantAndContext(q, v)
        qd = plant.MapVelocityToQDot(context, v*dt[0])
        # return q - qprev - qd
        '''
        As advised in
        https://stackoverflow.com/a/63510131/3177701
        '''
        ret_quat = q[0:4] - apply_angular_velocity_to_quaternion(qprev[0:4], v[0:3], dt[0])
        ret_linear = (q - qprev - qd)[4:]
        ret = np.hstack([ret_quat, ret_linear])
        # print(ret)
        return ret

    def eq7h(self, q_v_r):
        q, v, r = np.split(q_v_r, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        return  self.calc_r(q, v) - r

    def eq7i(self, q_v_ck):
        q, v, ck = np.split(q_v_ck, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        cj = np.reshape(ck, (self.num_contacts, 3))
        # print(f"q = {q}\nv={v}\nck={ck}")
        contact_positions = self.get_contact_positions(q, v).T
        return (contact_positions - cj).flatten()

    def eq8a_lhs(self, q_v_F):
        q, v, F = np.split(q_v_F, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        Fj = np.reshape(F, (self.num_contacts, 3))
        return [Fj[:,2].dot(self.get_contact_positions_z(q, v))] # Constraint functions must output vectors

    def eq8b_lhs(self, q_v_tau):
        q, v, tau = np.split(q_v_tau, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        tauj = self.toTauj(tau)
        return (tauj**2).T.dot(self.get_contact_positions_z(q, v)) # Outputs per axis sum of torques of all contact points

    def eq8c_2(self, q_v):
        q, v = np.split(q_v, [self.plant_float.num_positions()])
        return self.get_contact_positions_z(q, v)

    ''' Assume flat ground for now... '''
    def eq9a_lhs(self, F_c_cprev, i):
        F, c, c_prev = np.split(F_c_cprev, [
            self.contact_dim,
            self.contact_dim + self.contact_dim])
        Fj = np.reshape(F, (self.num_contacts, 3))
        cj = np.reshape(c, (self.num_contacts, 3))
        cj_prev = np.reshape(c_prev, (self.num_contacts, 3))
        return [Fj[i,2] * (cj[i] - cj_prev[i]).dot(np.array([1.0, 0.0, 0.0]))]

    def eq9b_lhs(self, F_c_cprev, i):
        F, c, c_prev = np.split(F_c_cprev, [
            self.contact_dim,
            self.contact_dim + self.contact_dim])
        Fj = np.reshape(F, (self.num_contacts, 3))
        cj = np.reshape(c, (self.num_contacts, 3))
        cj_prev = np.reshape(c_prev, (self.num_contacts, 3))
        return [Fj[i,2] * (cj[i] - cj_prev[i]).dot(np.array([0.0, 1.0, 0.0]))]

    def pose_error_cost(self, q_v_dt):
        q, v, dt = np.split(q_v_dt, [
            self.plant_float.num_positions(),
            self.plant_float.num_positions() + self.plant_float.num_velocities()])
        plant, context = self.getPlantAndContext(q, v)
        Q_q = 1.0 * np.identity(plant.num_velocities())
        q_err = plant.MapQDotToVelocity(context, q-self.q_nom)
        return (dt*(q_err.dot(Q_q).dot(q_err)))[0] # AddCost requires cost function to return scalar, not array

    def calcTrajectory(self, q_init, q_final, num_knot_points, max_time, pelvis_only=False):
        N = num_knot_points
        T = max_time

        sorted_joint_position_lower_limits = np.array([entry[1].lower for entry in getSortedJointLimits(self.plant_float)])
        sorted_joint_position_upper_limits = np.array([entry[1].upper for entry in getSortedJointLimits(self.plant_float)])
        sorted_joint_velocity_limits = np.array([entry[1].velocity for entry in getSortedJointLimits(self.plant_float)])

        prog = MathematicalProgram()
        q = prog.NewContinuousVariables(rows=N, cols=self.plant_float.num_positions(), name="q")
        v = prog.NewContinuousVariables(rows=N, cols=self.plant_float.num_velocities(), name="v")
        dt = prog.NewContinuousVariables(N, name="dt")
        r = prog.NewContinuousVariables(rows=N, cols=3, name="r")
        rd = prog.NewContinuousVariables(rows=N, cols=3, name="rd")
        rdd = prog.NewContinuousVariables(rows=N, cols=3, name="rdd")
        # The cols are ordered as
        # [contact1_x, contact1_y, contact1_z, contact2_x, contact2_y, contact2_z, ...]
        c = prog.NewContinuousVariables(rows=N, cols=self.contact_dim, name="c")
        F = prog.NewContinuousVariables(rows=N, cols=self.contact_dim, name="F")
        tau = prog.NewContinuousVariables(rows=N, cols=self.num_contacts, name="tau") # We assume only z torque exists
        h = prog.NewContinuousVariables(rows=N, cols=3, name="h")
        hd = prog.NewContinuousVariables(rows=N, cols=3, name="hd")

        ''' Additional variables not explicitly stated '''
        # Friction cone scale
        beta = prog.NewContinuousVariables(rows=N, cols=self.num_contacts*N_d, name="beta")

        g = np.array([0, 0, -Atlas.g])
        for k in range(N):
            ''' Eq(7a) '''
            Fj = np.reshape(F[k], (self.num_contacts, 3))
            (prog.AddLinearConstraint(eq(Atlas.M*rdd[k], np.sum(Fj, axis=0) + Atlas.M*g))
                    .evaluator().set_description(f"Eq(7a)[{k}]"))
            ''' Eq(7b) '''
            cj = np.reshape(c[k], (self.num_contacts, 3))
            tauj = self.toTauj(tau[k])
            (prog.AddConstraint(eq(hd[k], np.sum(np.cross(cj - r[k], Fj) + tauj, axis=0)))
                    .evaluator().set_description(f"Eq(7b)[{k}]"))
            ''' Eq(7c) '''
            (prog.AddConstraint(self.eq7c, lb=[0]*3, ub=[0]*3, vars=np.concatenate([q[k], v[k], h[k]]))
                .evaluator().set_description(f"Eq(7c)[{k}]"))
            ''' Eq(7h) '''
            # COM position has dimension 3
            (prog.AddConstraint(self.eq7h, lb=[0]*3, ub=[0]*3, vars=np.concatenate([q[k], v[k], r[k]]))
                    .evaluator().set_description(f"Eq(7h)[{k}]"))
            ''' Eq(7i) '''
            # np.concatenate cannot work q, cj since they have different dimensions
            (prog.AddConstraint(self.eq7i, lb=np.zeros(c[k].shape).flatten(), ub=np.zeros(c[k].shape).flatten(), vars=np.concatenate([q[k], v[k], c[k]]))
                    .evaluator().set_description(f"Eq(7i)[{k}]"))
            ''' Eq(7j) '''
            (prog.AddBoundingBoxConstraint([-10, -10, -MAX_GROUND_PENETRATION]*self.num_contacts, [10, 10, 10]*self.num_contacts, c[k])
                    .evaluator().set_description(f"Eq(7j)[{k}]"))
            ''' Eq(7k) '''
            ''' Constrain admissible posture '''
            (prog.AddBoundingBoxConstraint(sorted_joint_position_lower_limits, sorted_joint_position_upper_limits,
                    q[k, Atlas.FLOATING_BASE_QUAT_DOF:]).evaluator().set_description(f"Eq(7k)[{k}] joint position"))
            ''' Constrain velocities '''
            (prog.AddBoundingBoxConstraint(-sorted_joint_velocity_limits, sorted_joint_velocity_limits,
                v[k, Atlas.FLOATING_BASE_DOF:]).evaluator().set_description(f"Eq(7k)[{k}] joint velocity"))
            ''' Constrain forces within friction cone '''
            beta_k = np.reshape(beta[k], (self.num_contacts, N_d))
            for i in range(self.num_contacts):
                beta_v = beta_k[i].dot(self.friction_cone_components[:,i,:])
                (prog.AddLinearConstraint(eq(Fj[i], beta_v))
                        .evaluator().set_description(f"Eq(7k)[{k}] friction cone constraint[{i}]"))
            ''' Constrain beta positive '''
            for b in beta_k.flat:
                (prog.AddLinearConstraint(b >= 0.0)
                        .evaluator().set_description(f"Eq(7k)[{k}] beta >= 0 constraint"))
            ''' Constrain torques - assume torque linear to friction cone'''
            friction_torque_coefficient = 0.1
            for i in range(self.num_contacts):
                max_torque = friction_torque_coefficient * np.sum(beta_k[i])
                (prog.AddLinearConstraint(le(tau[k][i], np.array([max_torque])))
                        .evaluator().set_description(f"Eq(7k)[{k}] friction torque upper limit"))
                (prog.AddLinearConstraint(ge(tau[k][i], np.array([-max_torque])))
                        .evaluator().set_description(f"Eq(7k)[{k}] friction torque lower limit"))

            if ENABLE_COMPLEMENTARITY_CONSTRAINTS:
                ''' Eq(8a) '''
                (prog.AddConstraint(self.eq8a_lhs, lb=[-slack], ub=[slack], vars=np.concatenate([q[k], v[k], F[k]]))
                        .evaluator().set_description(f"Eq(8a)[{k}]"))
                ''' Eq(8b) '''
                (prog.AddConstraint(self.eq8b_lhs, lb=[-slack]*3, ub=[slack]*3, vars=np.concatenate([q[k], v[k], tau[k]]))
                        .evaluator().set_description(f"Eq(8b)[{k}]"))
                ''' Eq(8c) '''
                (prog.AddLinearConstraint(ge(Fj[:,2], 0.0))
                        .evaluator().set_description(f"Eq(8c)[{k}] contact force greater than zero"))
                # TODO: Why can't this be converted to a linear constraint?
                (prog.AddConstraint(self.eq8c_2, lb=[-MAX_GROUND_PENETRATION]*self.num_contacts, ub=[float('inf')]*self.num_contacts, vars=np.concatenate([q[k], v[k]]))
                        .evaluator().set_description(f"Eq(8c)[{k}] z position greater than zero"))

        for k in range(1, N):
            ''' Eq(7d) '''
            # dt[k] must be converted to an array
            (prog.AddConstraint(self.eq7d, lb=[0.0]*self.plant_float.num_positions(), ub=[0.0]*self.plant_float.num_positions(),
                    vars=np.concatenate([q[k], q[k-1], v[k], [dt[k]]]))
                .evaluator().set_description(f"Eq(7d)[{k}]"))

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
            # prog.AddConstraint(eq(q[k,0] - q[k-1,0], 0.5*(w1*q4 - w2*q3 + w3*q2)).reshape((1,)))
            # prog.AddConstraint(eq(q[k,1] - q[k-1,1], 0.5*(w1*q3 + w2*q4 - w3*q1)).reshape((1,)))
            # prog.AddConstraint(eq(q[k,2] - q[k-1,2], 0.5*(-w1*q2 + w2*q1 + w3*q4)).reshape((1,)))
            # prog.AddConstraint(eq(q[k,3] - q[k-1,3], 0.5*(-w1*q1 - w2*q2 - w3*q3)).reshape((1,)))
            # ''' Constrain other positions '''
            # prog.AddConstraint(eq(q[k, 4:] - q[k-1, 4:], v[k, 3:]*dt[k]))

            ''' Eq(7e) '''
            (prog.AddConstraint(eq(h[k] - h[k-1], hd[k]*dt[k]))
                .evaluator().set_description(f"Eq(7e)[{k}]"))
            ''' Eq(7f) '''
            (prog.AddConstraint(eq(r[k] - r[k-1], (rd[k] + rd[k-1])/2*dt[k]))
                .evaluator().set_description(f"Eq(7f)[{k}]"))
            ''' Eq(7g) '''
            (prog.AddConstraint(eq(rd[k] - rd[k-1], rdd[k]*dt[k]))
                .evaluator().set_description(f"Eq(7g)[{k}]"))

            Fj = np.reshape(F[k], (self.num_contacts, 3))
            cj = np.reshape(c[k], (self.num_contacts, 3))
            cj_prev = np.reshape(c[k-1], (self.num_contacts, 3))

            if ENABLE_COMPLEMENTARITY_CONSTRAINTS:
                for i in range(self.num_contacts):
                    ''' Eq(9a) '''
                    '''
                    i=i is used to capture the outer scope i variable
                    https://stackoverflow.com/a/2295372/3177701
                    '''
                    (prog.AddConstraint(lambda F_c_cprev, i=i : self.eq9a_lhs(F_c_cprev, i), ub=[slack], lb=[-slack], vars=np.concatenate([F[k], c[k], c[k-1]]))
                            .evaluator().set_description("Eq(9a)[{k}][{i}]"))
                    ''' Eq(9b) '''
                    '''
                    i=i is used to capture the outer scope i variable
                    https://stackoverflow.com/a/2295372/3177701
                    '''
                    (prog.AddConstraint(lambda F_c_cprev, i=i : self.eq9b_lhs(F_c_cprev, i), ub=[slack], lb=[-slack], vars=np.concatenate([F[k], c[k], c[k-1]]))
                            .evaluator().set_description("Eq(9b)[{k}][{i}]"))

        ''' Eq(10) '''
        for k in range(N):
            Q_v = 0.5 * np.identity(self.plant_float.num_velocities())
            prog.AddCost(self.pose_error_cost, vars=np.concatenate([q[k], v[k], [dt[k]]])) # np.concatenate requires items to have compatible shape
            prog.AddCost(dt[k]*(
                    + v[k].dot(Q_v).dot(v[k])
                    + rdd[k].dot(rdd[k])))

        ''' Additional constraints not explicitly stated '''
        ''' Constrain initial pose '''
        (prog.AddLinearConstraint(eq(q[0], q_init))
                .evaluator().set_description("initial pose"))
        ''' Constrain initial velocity '''
        (prog.AddLinearConstraint(eq(v[0], 0.0))
                .evaluator().set_description("initial velocity"))
        ''' Constrain final pose '''
        if pelvis_only:
            (prog.AddLinearConstraint(eq(q[-1, 0:7], q_final[0:7]))
                    .evaluator().set_description("final pose"))
        else:
            (prog.AddLinearConstraint(eq(q[-1], q_final))
                    .evaluator().set_description("final pose"))
        ''' Constrain final velocity '''
        (prog.AddLinearConstraint(eq(v[-1], 0.0))
                .evaluator().set_description("final velocity"))
        ''' Constrain final COM velocity '''
        (prog.AddLinearConstraint(eq(rd[-1], 0.0))
                .evaluator().set_description("final COM velocity"))
        ''' Constrain final COM acceleration '''
        (prog.AddLinearConstraint(eq(rdd[-1], 0.0))
                .evaluator().set_description("final COM acceleration"))
        ''' Constrain time taken '''
        (prog.AddLinearConstraint(np.sum(dt) <= T)
                .evaluator().set_description("max time"))
        ''' Constrain first time step '''
        # Note that the first time step is only used in the initial cost calculation
        # and not in the backwards Euler
        (prog.AddLinearConstraint(dt[0] == 0)
                .evaluator().set_description("first timestep"))
        ''' Constrain remaining time step '''
        # Values taken from
        # https://colab.research.google.com/github/RussTedrake/underactuated/blob/master/exercises/simple_legs/compass_gait_limit_cycle/compass_gait_limit_cycle.ipynb
        (prog.AddLinearConstraint(ge(dt[1:], [0.005]*(N-1)))
                .evaluator().set_description("min timestep"))
        (prog.AddLinearConstraint(le(dt, [0.05]*N))
                .evaluator().set_description("max timestep"))
        ''' Constrain max joint acceleration '''
        for k in range(1, N):
            (prog.AddLinearConstraint(ge((v[k] - v[k-1]), -MAX_JOINT_ACCELERATION*dt[k]))
                    .evaluator().set_description(f"min joint acceleration[{k}]"))
            (prog.AddLinearConstraint(le((v[k] - v[k-1]), MAX_JOINT_ACCELERATION*dt[k]))
                    .evaluator().set_description(f"max joint acceleration[{k}]"))
        '''
        Constrain unbounded variables to improve IPOPT performance
        because IPOPT is an interior point method which works poorly for unbounded variables
        '''
        (prog.AddLinearConstraint(le(F.flatten(), np.ones(F.shape).flatten()*1e3))
                .evaluator().set_description("max F"))
        (prog.AddBoundingBoxConstraint(-1e3, 1e3, tau)
                .evaluator().set_description("bound tau"))
        (prog.AddLinearConstraint(le(beta.flatten(), np.ones(beta.shape).flatten()*1e3))
                .evaluator().set_description("max beta"))

        ''' Solve '''
        initial_guess = np.empty(prog.num_vars())
        dt_guess = [0.0] + [T/(N-1)] * (N-1)
        prog.SetDecisionVariableValueInVector(dt, dt_guess, initial_guess)
        # Guess q to avoid initializing with invalid quaternion
        quat_traj_guess = PiecewiseQuaternionSlerp()
        quat_traj_guess.Append(0, Quaternion(q_init[0:4]))
        quat_traj_guess.Append(T, Quaternion(q_final[0:4]))
        position_traj_guess = PiecewisePolynomial.FirstOrderHold([0.0, T], np.vstack([q_init[4:], q_final[4:]]).T)
        q_guess = np.array([np.hstack([
            Quaternion(quat_traj_guess.value(t)).wxyz(), position_traj_guess.value(t).flatten()])
            for t in np.linspace(0, T, N)])
        prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)

        v_traj_guess = position_traj_guess.MakeDerivative()
        w_traj_guess = quat_traj_guess.MakeDerivative()
        v_guess = np.array([
            np.hstack([w_traj_guess.value(t).flatten(), v_traj_guess.value(t).flatten()])
            for t in np.linspace(0, T, N)])
        prog.SetDecisionVariableValueInVector(v, v_guess, initial_guess)

        c_guess = np.array([
            self.get_contact_positions(q_guess[i], v_guess[i]).T.flatten() for i in range(N)])
        for i in range(N):
            assert((self.eq7i(np.concatenate([q_guess[i], v_guess[i], c_guess[i]])) == 0.0).all())
        prog.SetDecisionVariableValueInVector(c, c_guess, initial_guess)

        r_guess = np.array([
            self.calc_r(q_guess[i], v_guess[i]) for i in range(N)])
        prog.SetDecisionVariableValueInVector(r, r_guess, initial_guess)

        h_guess = np.array([
            self.calc_h(q_guess[i], v_guess[i]) for i in range(N)])
        prog.SetDecisionVariableValueInVector(h, h_guess, initial_guess)

        solver = SnoptSolver()
        options = SolverOptions()
        # options.SetOption(solver.solver_id(), "max_iter", 50000)
        # This doesn't seem to do anything...
        # options.SetOption(CommonSolverOption.kPrintToConsole, True)
        start_solve_time = time.time()
        print(f"Start solving...")
        result = solver.Solve(prog, initial_guess, options) # Currently takes around 30 mins
        print(f"Solve time: {time.time() - start_solve_time}s  Cost: {result.get_optimal_cost()} Success: {result.is_success()}")
        self.q_sol = result.GetSolution(q)
        self.v_sol = result.GetSolution(v)
        self.dt_sol = result.GetSolution(dt)
        self.r_sol = result.GetSolution(r)
        self.rd_sol = result.GetSolution(rd)
        self.rdd_sol = result.GetSolution(rdd)
        self.c_sol = result.GetSolution(c)
        self.F_sol = result.GetSolution(F)
        self.tau_sol = result.GetSolution(tau)
        self.h_sol = result.GetSolution(h)
        self.hd_sol = result.GetSolution(hd)
        self.beta_sol = result.GetSolution(beta)
        if not result.is_success():
            print(result.GetInfeasibleConstraintNames(prog))
            pdb.set_trace()

        return (self.r_sol, self.rd_sol, self.rdd_sol,
                self.q_sol, self.v_sol, self.dt_sol)

def main():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(mbp_time_step))
    load_atlas(plant, add_ground=True)
    plant_context = plant.CreateDefaultContext()

    q_init = plant.GetPositions(plant_context)
    q_init[6] = 0.95 # Avoid initializing with ground penetration
    q_final = q_init.copy()
    q_final[4] = 0.5 # x position of pelvis
    q_final[6] = 0.90 # z position of pelvis (to make sure final pose touches ground)
    upright_context = plant.CreateDefaultContext()
    set_atlas_initial_pose(plant, upright_context)
    q_nom = plant.GetPositions(upright_context)

    num_knot_points = 30
    max_time = 1.0

    export_filename = f"sample(final_x_{q_final[4]})(num_knot_points_{num_knot_points})(max_time_{max_time})"

    planner = HumanoidPlanner(plant, Atlas.CONTACTS_PER_FRAME, q_nom)
    if not PLAYBACK_ONLY:
        print(f"Starting pos: {q_init}\nFinal pos: {q_final}")
        r_traj, rd_traj, rdd_traj, q_traj, v_traj, dt_traj = (
                planner.calcTrajectory(q_init, q_final, num_knot_points, max_time, pelvis_only=True))

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
