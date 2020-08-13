#!/usr/bin/python3

# This implements the paper
# An Efficiently Solvable Quadratic Program for Stabilizing Dynamic Locomotion
# by Scott Kuindersma, Frank Permenter, and Russ Tedrake

# Notes
# Atlas is 175kg according to drake/share/drake/examples/atlas/urdf/atlas_convex_hull.urdf

# TODO:
# Convert to time-varying y_desired and z_com

from load_atlas import load_atlas, set_atlas_initial_pose
import numpy as np
from pydrake.systems.framework import DiagramBuilder
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.systems.controllers import LinearQuadraticRegulator
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.geometry import ConnectDrakeVisualizer, SceneGraph
from pydrake.multibody.plant import ConnectContactResultsToDrakeVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import BasicVector, LeafSystem
from pydrake.common.eigen_geometry import Quaternion

from collections import namedtuple

import time
import pdb

FLOATING_BASE_DOF = 6
FLOATING_BASE_QUAT_DOF = 7 # Start index of actuated joints in generalized positions
NUM_ACTUATED_DOF = 30
TOTAL_DOF = FLOATING_BASE_DOF + NUM_ACTUATED_DOF
g = 9.81
z_com = 1.220 # COM after 0.05s
com_state_size = 4
half_com_state_size = int(com_state_size/2.0)
zmp_state_size = 2
mbp_time_step = 1.0e-3

JointLimit = namedtuple("JointLimit", ["effort", "lower", "upper"])

JOINT_LIMITS = {
        "back_bkx" : JointLimit(300, -0.523599, 0.523599),
        "back_bky" : JointLimit(445, -0.219388, 0.538783),
        "back_bkz" : JointLimit(106, -0.663225, 0.663225),
        "l_arm_elx": JointLimit(112, 0, 2.35619),
        "l_arm_ely": JointLimit(63,  0, 3.14159),
        "l_arm_shx": JointLimit(99, -1.5708, 1.5708),
        "l_arm_shz": JointLimit(87, -1.5708, 0.785398),
        "l_arm_mwx": JointLimit(25, -1.7628, 1.7628),
        "l_arm_uwy": JointLimit(25, -3.011, 3.011),
        "l_arm_lwy": JointLimit(25, -2.9671, 2.9671),
        "l_leg_akx": JointLimit(360, -0.8, 0.8),
        "l_leg_aky": JointLimit(740, -1, 0.7),
        "l_leg_hpx": JointLimit(530, -0.523599, 0.523599),
        "l_leg_hpy": JointLimit(840, -1.61234, 0.65764),
        "l_leg_hpz": JointLimit(275, -0.174358, 0.786794),
        "l_leg_kny": JointLimit(890, 0,  2.35637),
        "neck_ay"  : JointLimit(25, -0.602139, 1.14319),
        "r_arm_elx": JointLimit(112, -2.35619, 0),
        "r_arm_ely": JointLimit(63,  0,  3.14159),
        "r_arm_shx": JointLimit(99, -1.5708, 1.5708),
        "r_arm_shz": JointLimit(87, -0.785398, 1.5708),
        "r_arm_mwx": JointLimit(25, -1.7628, 1.7628),
        "r_arm_uwy": JointLimit(25, -3.011, 3.011),
        "r_arm_lwy": JointLimit(25, -2.9671, 2.9671),
        "r_leg_akx": JointLimit(360, -0.8, 0.8),
        "r_leg_aky": JointLimit(740, -1, 0.7),
        "r_leg_hpx": JointLimit(530, -0.523599, 0.523599),
        "r_leg_hpy": JointLimit(840, -1.61234, 0.65764),
        "r_leg_hpz": JointLimit(275, -0.786794, 0.174358),
        "r_leg_kny": JointLimit(890, 0, 2.35637)
}

tau_min = -200.0
tau_max = 200.0
mu = 1.0 # Coefficient of friction, same as in load_atlas.py
eta_min = -0.2
eta_max = 0.2

# Taken from drake/drake-build/install/share/drake/examples/atlas/urdf/atlas_minimal_contact.urdf
lfoot_full_contact_points = np.array([
    [-0.0876,0.066,-0.07645], # left heel
    [-0.0876,-0.0626,-0.07645], # right heel
    [0.1728,0.066,-0.07645], # left toe
    [0.1728,-0.0626,-0.07645], # right toe
    [0.086,0.066,-0.07645], # left midfoot_front
    [0.086,-0.0626,-0.07645], # right midfoot_front
    [-0.0008,0.066,-0.07645], # left midfoot_rear
    [-0.0008,-0.0626,-0.07645] # right midfoot_rear
]).T

rfoot_full_contact_points = np.array([
    [-0.0876,0.0626,-0.07645], # left heel
    [-0.0876,-0.066,-0.07645], # right heel
    [0.1728,0.0626,-0.07645], # left toe
    [0.1728,-0.066,-0.07645], # right toe
    [0.086,0.0626,-0.07645], # left midfoot_front
    [0.086,-0.066,-0.07645], # right midfoot_front
    [-0.0008,0.0626,-0.07645], # left midfoot_rear
    [-0.0008,-0.066,-0.07645] # right midfoot_rear
]).T
N_d = 4 # friction cone approximated as a i-pyramid
N_f = 3 # contact force dimension

def normalize(q):
    return q / np.linalg.norm(q)

# From http://www.nt.ntnu.no/users/skoge/prost/proceedings/ecc-2013/data/papers/0927.pdf
def calcAngularError(q_target, q_source):
    try:
        quat_err = Quaternion(normalize(q_source)).multiply(Quaternion(normalize(q_target)).inverse())
        if quat_err.w() < 0:
            return -quat_err.xyz()
        else:
            return quat_err.xyz()
    except:
        pdb.set_trace()

def calcPoseError(target, source):
    assert(target.size == source.size)
    # Make sure pose is expressed in generalized positions (quaternion base)
    assert(target.size == NUM_ACTUATED_DOF + FLOATING_BASE_QUAT_DOF)

    error = np.zeros(target.shape[0]-1)
    error[0:3] = calcAngularError(target[0:4], source[0:4])
    error[3:] = (source - target)[4:]
    return error

class HumanoidController(LeafSystem):
    def __init__(self):
        self.start_time = None
        LeafSystem.__init__(self)
        self.plant = MultibodyPlant(mbp_time_step)
        load_atlas(self.plant)
        self.upright_context = self.plant.CreateDefaultContext()
        set_atlas_initial_pose(self.plant, self.upright_context)
        self.q_des = self.plant.GetPositions(self.upright_context)

        # Assume y_desired is fixed for now
        # self.input_y_desired_idx = self.DeclareVectorInputPort("y_des", BasicVector(zmp_state_size)).get_index()
        y_des = np.array([0.1, 0.0])
        self.input_q_v_idx = self.DeclareVectorInputPort("q_v",
                BasicVector(self.plant.GetPositions(self.upright_context).size + self.plant.GetVelocities(self.upright_context).size)).get_index()
        self.output_tau_idx = self.DeclareVectorOutputPort("tau", BasicVector(NUM_ACTUATED_DOF), self.calcTorqueOutput).get_index()

        ## Eq(1)
        A = np.vstack([
            np.hstack([0*np.identity(half_com_state_size), 1*np.identity(half_com_state_size)]),
            np.hstack([0*np.identity(half_com_state_size), 0*np.identity(half_com_state_size)])])
        self.A = A
        B_1 = np.vstack([
            0*np.identity(half_com_state_size),
            1*np.identity(half_com_state_size)])
        self.B_1 = B_1

        ## Eq(4)
        C_2 = np.hstack([np.identity(2), np.zeros((2,2))]) # C in Eq(2)
        self.C_2 = C_2
        D = -z_com / g * np.identity(zmp_state_size)
        self.D = D
        Q = 1.0 * np.identity(zmp_state_size)
        self.Q = Q

        ## Eq(6)
        # y.T*Q*y
        # = (C*x+D*u)*Q*(C*x+D*u).T
        # = x.T*C.T*Q*C*x + u.T*D.T*Q*D*u + x.T*C.T*Q*D*u + u.T*D.T*Q*C*X
        # = ..                            + 2*x.T*C.T*Q*D*u
        K, S = LinearQuadraticRegulator(A, B_1, C_2.T.dot(Q).dot(C_2), D.T.dot(Q).dot(D), C_2.T.dot(Q).dot(D))
        self.K = K
        self.S = S
        def V(x, u): # Assume constant z_com, we don't need tvLQR
            y = C_2.dot(x) + D.dot(u)
            def dJ_dx(x):
                return x.T.dot(S.T+S) # https://math.stackexchange.com/questions/20694/vector-derivative-w-r-t-its-transpose-fracdaxdxt
            y_bar = y - y_des
            x_bar = x - np.hstack([y_des, [0.0, 0.0]])
            xd_bar = A.dot(x_bar) + B_1.dot(u)
            return y_bar.T.dot(Q).dot(y_bar) + dJ_dx(x_bar).dot(xd_bar)
        self.V = V

    def create_qp1(self, plant_context):
        # Determine contact points
        lfoot_full_contact_pos = self.plant.CalcPointsPositions(plant_context, self.plant.GetFrameByName("l_foot"),
                lfoot_full_contact_points, self.plant.world_frame())
        lfoot_contact_points = lfoot_full_contact_points[:, np.where(lfoot_full_contact_pos[2,:] <= 0.0)[0]]
        rfoot_full_contact_pos = self.plant.CalcPointsPositions(plant_context, self.plant.GetFrameByName("r_foot"),
                rfoot_full_contact_points, self.plant.world_frame())
        rfoot_contact_points = rfoot_full_contact_points[:, np.where(rfoot_full_contact_pos[2,:] <= 0.0)[0]]
        # print("lfoot contact z pos: " + str(lfoot_full_contact_pos[2]))
        # print("lfoot # contacts: " + str(lfoot_contact_points.shape[1]))
        # print("rfoot contact z pos: " + str(rfoot_full_contact_pos[2]))
        # print("rfoot # contacts: " + str(rfoot_contact_points.shape[1]))
        N_c_lfoot = lfoot_contact_points.shape[1] # Num contacts per foot
        N_c_rfoot = rfoot_contact_points.shape[1] # Num contacts per foot
        N_c = N_c_lfoot + N_c_rfoot # num contact points

        ## Eq(7)
        H = self.plant.CalcMassMatrixViaInverseDynamics(plant_context)
        self.H = H
        # Note that CalcGravityGeneralizedForces assumes the form Mv̇ + C(q, v)v = tau_g(q) + tau_app
        # while Eq(7) assumes gravity is accounted in C (on the left hand side)
        C_7 = self.plant.CalcBiasTerm(plant_context) - self.plant.CalcGravityGeneralizedForces(plant_context)
        self.C_7 = C_7
        B_7 = self.plant.MakeActuationMatrix()
        self.B_7 = B_7

        Phi_lfoot = self.plant.CalcJacobianTranslationalVelocity(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("l_foot"),
                lfoot_contact_points, self.plant.world_frame(), self.plant.world_frame())

        Phi_rfoot = self.plant.CalcJacobianTranslationalVelocity(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("r_foot"),
                rfoot_contact_points, self.plant.world_frame(), self.plant.world_frame())
        Phi = np.vstack([Phi_lfoot, Phi_rfoot])
        self.Phi = Phi

        ## Eq(8)
        # From np.sort(np.nonzero(B_7)[0]) we know that indices 0-5 are the unactuated 6 DOF floating base and 6-35 are the actuated 30 DOF robot joints
        v_idx_act = 6 # Start index of actuated joints in generalized velocities
        H_f = H[0:v_idx_act,:]
        self.H_f = H_f
        H_a = H[v_idx_act:,:]
        self.H_a = H_a
        C_f = C_7[0:v_idx_act]
        self.C_f = C_f
        C_a = C_7[v_idx_act:]
        self.C_a = C_a
        B_a = B_7[v_idx_act:,:]
        self.B_a = B_a
        Phi_f_T = Phi.T[0:v_idx_act:,:]
        self.Phi_f_T = Phi_f_T
        Phi_a_T = Phi.T[v_idx_act:,:]
        self.Phi_a_T = Phi_a_T

        ## Eq(9)
        # Assume flat ground for now
        n = np.array([
            [0],
            [0],
            [1.0]])
        d = np.array([
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0]])
        v = np.zeros((N_d, N_c, N_f))
        for i in range(N_d):
            for j in range(N_c):
                v[i,j] = (n+mu*d)[:,i]

        ## Quadratic Program I
        prog = MathematicalProgram()
        qdd = prog.NewContinuousVariables(self.plant.num_velocities(), name="qdd") # To ignore 6 DOF floating base
        self.qdd = qdd
        beta = prog.NewContinuousVariables(N_d,N_c, name="beta")
        self.beta = beta
        lambd = prog.NewContinuousVariables(N_f*N_c, name="lambda")
        self.lambd = lambd

        # Jacobians ignoring the 6DOF floating base
        J_lfoot = np.zeros((N_f*N_c_lfoot, TOTAL_DOF))
        for i in range(N_c_lfoot):
            J_lfoot[N_f*i:N_f*(i+1),:] = self.plant.CalcJacobianSpatialVelocity(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("l_foot"),
                lfoot_contact_points[:,i], self.plant.world_frame(), self.plant.world_frame())[3:]
        J_rfoot = np.zeros((N_f*N_c_rfoot, TOTAL_DOF))
        for i in range(N_c_rfoot):
            J_rfoot[N_f*i:N_f*(i+1),:] = self.plant.CalcJacobianSpatialVelocity(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("r_foot"),
                rfoot_contact_points[:,i], self.plant.world_frame(), self.plant.world_frame())[3:]

        J = np.vstack([J_lfoot, J_rfoot])
        self.J = J
        assert(J.shape == (N_c*N_f, TOTAL_DOF))

        eta = prog.NewContinuousVariables(J.shape[0], name="eta")
        self.eta = eta

        x = prog.NewContinuousVariables(com_state_size, name="x") # x_com, y_com, x_com_d, y_com_d
        self.x = x
        u = prog.NewContinuousVariables(half_com_state_size, name="u") # x_com_dd, y_com_dd
        self.u = u

        ## Eq(10)
        w = 0.01
        epsilon = 1.0e-8
        K_p = 10.0
        K_d = 2.0
        frame_weights = np.ones((TOTAL_DOF))

        # For generalized positions, first 7 values are 4 quaternion + 3 x,y,z
        q = self.plant.GetPositions(plant_context)
        # For generalized velocities, first 6 values are 3 rotational velocities + 3 xd, yd, zd
        # Hence this not strictly the derivative of q
        qd = self.plant.GetVelocities(plant_context)

        # Convert q, q_des to generalized velocities form
        q_err = calcPoseError(self.q_des, q)
        print(f"Pelvis error: {q_err[0:3]}")
        ## FIXME: Not sure if it's a good idea to ignore the x, y, z position of pelvis
        # ignored_pose_indices = {3, 4, 5} # Ignore x position, y position
        ignored_pose_indices = {} # Ignore x position, y position
        relevant_pose_indices = list(set(range(TOTAL_DOF)) - set(ignored_pose_indices))
        self.relevant_pose_indices = relevant_pose_indices
        qdd_des = -K_p*q_err - K_d*qd
        self.qdd_des = qdd_des
        qdd_err = qdd_des - qdd
        qdd_err = qdd_err*frame_weights
        qdd_err = qdd_err[relevant_pose_indices]
        prog.AddCost(
                self.V(x, u)
                + w*((qdd_err).dot(qdd_err))
                + epsilon * np.sum(np.square(beta))
                + eta.dot(eta))

        ## Eq(11)
        eq11_lhs = H_f.dot(qdd)+C_f
        self.eq11_lhs = eq11_lhs
        eq11_rhs = Phi_f_T.dot(lambd)
        self.eq11_rhs = eq11_rhs
        for i in range(eq11_lhs.size):
            prog.AddConstraint(eq11_lhs[i] == eq11_rhs[i])

        ## Eq(12)
        alpha = 0.1
        Jd_qd_lfoot = self.plant.CalcBiasTranslationalAcceleration(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("l_foot"),
                lfoot_contact_points, self.plant.world_frame(), self.plant.world_frame())
        Jd_qd_rfoot = self.plant.CalcBiasTranslationalAcceleration(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("r_foot"),
                rfoot_contact_points, self.plant.world_frame(), self.plant.world_frame())
        Jd_qd = np.concatenate([Jd_qd_lfoot.flatten(), Jd_qd_rfoot.flatten()])
        assert(Jd_qd.shape == (N_c*3,))
        eq12_lhs = J.dot(qdd) + Jd_qd
        self.eq12_lhs = eq12_lhs
        eq12_rhs = -alpha*J.dot(qd) + eta
        self.eq12_rhs = eq12_rhs
        for i in range(eq12_lhs.shape[0]):
            prog.AddConstraint(eq12_lhs[i] == eq12_rhs[i])

        ## Eq(13)
        def tau(qdd, lambd):
            return np.linalg.inv(B_a).dot(H_a.dot(qdd) + C_a - Phi_a_T.dot(lambd))
        self.tau = tau
        eq13_lhs = self.tau(qdd, lambd)
        self.eq13_lhs = eq13_lhs
        for name, limit in JOINT_LIMITS.items():
            i = self.getActuatorIndex(name)
            # eq13_lhs is already expressed in actuator space
            prog.AddConstraint(eq13_lhs[i] >= -limit.effort)
            prog.AddConstraint(eq13_lhs[i] <= limit.effort)

        ## Eq(14)
        for j in range(N_c):
            beta_v = beta[:,j].dot(v[:,j])
            for k in range(N_f):
                prog.AddConstraint(lambd[N_f*j+k] == beta_v[k])

        ## Eq(15)
        for b in beta.flat:
            prog.AddConstraint(b >= 0.0)

        ## Eq(16)
        for i in range(eta.shape[0]):
            prog.AddConstraint(eta[i] >= eta_min)
            prog.AddConstraint(eta[i] <= eta_max)

        ### Below are constraints that aren't explicitly stated in the paper
        ### but that seemed important

        ## Enforce x as com
        com = self.plant.CalcCenterOfMassPosition(plant_context)
        com_d = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                plant_context, JacobianWrtVariable.kV,
                self.plant.world_frame(), self.plant.world_frame()).dot(qd)
        prog.AddConstraint(x[0] == com[0])
        prog.AddConstraint(x[1] == com[1])
        prog.AddConstraint(x[2] == com_d[0])
        prog.AddConstraint(x[3] == com_d[1])

        ## Enforce u as com_dd
        com_dd = (
                self.plant.CalcBiasCenterOfMassTranslationalAcceleration(
                    plant_context, JacobianWrtVariable.kV,
                    self.plant.world_frame(), self.plant.world_frame())
                + self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                    plant_context, JacobianWrtVariable.kV,
                    self.plant.world_frame(), self.plant.world_frame())
                .dot(qdd))
        prog.AddConstraint(u[0] == com_dd[0])
        prog.AddConstraint(u[1] == com_dd[1])

        ## Respect joint limits
        for name, limit in JOINT_LIMITS.items():
            # Get the corresponding joint value
            joint_pos = self.plant.GetJointByName(name).get_angle(plant_context)
            # Get the corresponding actuator index
            act_idx = self.getActuatorIndex(name)
            # Use the actuator index to find the corresponding generalized coordinate index
            q_idx = np.where(B_7[:,act_idx] == 1)[0][0]

            if joint_pos >= limit.upper:
                # print(f"Joint {name} max reached")
                prog.AddConstraint(qdd[q_idx] <= 0.0)
            elif joint_pos <= limit.lower:
                # print(f"Joint {name} min reached")
                prog.AddConstraint(qdd[q_idx] >= 0.0)

        return prog

    def calcTorqueOutput(self, context, output):
        if not self.start_time:
            self.start_time = context.get_time()

        ## Start controller only after foot makes contact with ground
        if context.get_time() - self.start_time < 0.02:
            output.SetFromVector(np.zeros(30))
            return

        q_v = self.EvalVectorInput(context, self.input_q_v_idx).get_value()
        current_plant_context = self.plant.CreateDefaultContext()
        self.plant.SetPositionsAndVelocities(current_plant_context, q_v)
        prog = self.create_qp1(current_plant_context)
        result = Solve(prog)
        if not result.is_success():
            print(f"FAILED")
            pdb.set_trace()
            exit(-1)
        print(f"Cost: {result.get_optimal_cost()}")
        qdd_sol = result.GetSolution(self.qdd)
        lambd_sol = result.GetSolution(self.lambd)
        x_sol = result.GetSolution(self.x)
        u_sol = result.GetSolution(self.u)
        beta_sol = result.GetSolution(self.beta)
        eta_sol = result.GetSolution(self.eta)

        com, comd, comdd = self.calcCOM(current_plant_context, result)

        # print(f"comdd z: {comdd[2]}")
        # self.plant.EvalBodyPoseInWorld(current_plant_context, self.plant.GetBodyByName("pelvis")).rotation().ToQuaternion().xyz()
        print(f"pelvis angular position = {Quaternion(normalize(q_v[0:4])).xyz()}")
        print(f"pelvis angular velocity = {q_v[FLOATING_BASE_QUAT_DOF + NUM_ACTUATED_DOF:FLOATING_BASE_QUAT_DOF + NUM_ACTUATED_DOF + 3]}")
        print(f"pelvis angular acceleration = {qdd_sol[0:3]}")
        print(f"pelvis translational position = {q_v[4:7]}")
        print(f"pelvis translational velocity = {q_v[FLOATING_BASE_QUAT_DOF + NUM_ACTUATED_DOF + 3 : FLOATING_BASE_QUAT_DOF + NUM_ACTUATED_DOF + 6]}")
        print(f"pelvis translational acceleration = {qdd_sol[3:6]}")
        # print(f"beta = {beta_sol}")
        # print(f"lambda z = {lambd_sol[2::3]}")
        print(f"Total force z = {np.sum(lambd_sol[2::3])}")

        tau = self.tau(qdd_sol, lambd_sol)
        interested_joints = [
                "back_bky",
                "l_leg_hpy",
                "r_leg_hpy"
        ]
        print(f"tau = {tau[self.getActuatorIndices(interested_joints)]}")
        print(f"joint angles = {self.getJointValues(interested_joints, current_plant_context)}")

        output.SetFromVector(tau)
        print("========================================")

    def calcCOM(self, current_plant_context, result):
        qdd_sol = result.GetSolution(self.qdd)
        qd = self.plant.GetVelocities(current_plant_context)
        com = self.plant.CalcCenterOfMassPosition(current_plant_context)
        com_d = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                current_plant_context, JacobianWrtVariable.kV,
                self.plant.world_frame(), self.plant.world_frame()).dot(qd)
        com_dd = (
                self.plant.CalcBiasCenterOfMassTranslationalAcceleration(
                    current_plant_context, JacobianWrtVariable.kV,
                    self.plant.world_frame(), self.plant.world_frame())
                + self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                    current_plant_context, JacobianWrtVariable.kV,
                    self.plant.world_frame(), self.plant.world_frame())
                .dot(qdd_sol))
        return com, com_d, com_dd

    def getActuatorIndex(self, joint_name):
        return int(self.plant.GetJointActuatorByName(joint_name + "_motor").index())

    def getActuatorIndices(self, joint_names):
        ret = []
        for name in joint_names:
            idx = self.getActuatorIndex(name)
            ret.append(idx)
        return ret

    def getJointValues(self, joint_names, context):
        ret = []
        for name in joint_names:
            ret.append(self.plant.GetJointByName(name).get_angle(context))
        return ret

    def getOrderedJointLimits(self):
        ret = [None] * len(JOINT_LIMITS)
        for name, limit in JOINT_LIMITS.items():
            i = self.getActuatorIndex(name)
            ret[i] = limit
        return ret

def main():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(mbp_time_step))
    load_atlas(plant, add_ground=True)
    plant_context = plant.CreateDefaultContext()

    controller = builder.AddSystem(HumanoidController())
    controller.set_name("HumanoidController")

    builder.Connect(plant.get_state_output_port(), controller.GetInputPort("q_v"))
    builder.Connect(controller.GetOutputPort("tau"), plant.get_actuation_input_port())

    ConnectContactResultsToDrakeVisualizer(builder=builder, plant=plant)
    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    set_atlas_initial_pose(plant, plant_context)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(0.1)
    simulator.AdvanceTo(5.0)

if __name__ == "__main__":
    main()
