#!/usr/bin/python3

# This implements the paper
# An Efficiently Solvable Quadratic Program for Stabilizing Dynamic Locomotion
# by Scott Kuindersma, Frank Permenter, and Russ Tedrake

# TODO:
# Convert plant_context to be dependent on the time varying state via plant output ports
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

import time
import pdb

FLOATING_BASE_DOF = 6
NUM_ACTUATED_DOF = 30
TOTAL_DOF = FLOATING_BASE_DOF + NUM_ACTUATED_DOF
g = 9.81
z_com = 1.2 # Obtained experimentally
com_state_size = 4
half_com_state_size = int(com_state_size/2.0)
zmp_state_size = 2
mbp_time_step = 1.0e-3
tau_min = -200.0
tau_max = 200.0
mu = 1.0 # Coefficient of friction
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

v_idx_act = 6 # Start index of actuated joints in generalized velocities
q_idx_act = 7 # Start index of actuated joints in generalized positions

def normalize(q):
    return q / np.linalg.norm(q)

# From http://www.nt.ntnu.no/users/skoge/prost/proceedings/ecc-2013/data/papers/0927.pdf
def calcAngularError(q_target, q_source):
    try:
        quat_err = Quaternion(normalize(q_target)).multiply(Quaternion(normalize(q_source)).inverse())
        if quat_err.w() < 0:
            return -quat_err.xyz()
        else:
            return quat_err.xyz()
    except:
        pdb.set_trace()

def calcPoseError(target, source):
    assert(target.size == source.size)
    # Make sure pose is expressed in generalized positions (quaternion base)
    assert(target.size == NUM_ACTUATED_DOF + q_idx_act)

    correction = np.zeros(target.shape[0]-1)
    correction[0:3] = calcAngularError(target[0:4], source[0:4])
    correction[3:] = (target - source)[4:]
    return correction

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
        # self.input_y_desired_idx = self.DeclareVectorInputPort("yd", BasicVector(zmp_state_size)).get_index()
        self.input_q_v_idx = self.DeclareVectorInputPort("q_v",
                BasicVector(self.plant.GetPositions(self.upright_context).size + self.plant.GetVelocities(self.upright_context).size)).get_index()
        self.output_tau_idx = self.DeclareVectorOutputPort("tau", BasicVector(NUM_ACTUATED_DOF), self.calcTorqueOutput).get_index()

        ## Eq(1)
        A = np.vstack([
            np.hstack([0*np.identity(half_com_state_size), 1*np.identity(half_com_state_size)]),
            np.hstack([0*np.identity(half_com_state_size), 0*np.identity(half_com_state_size)])])
        B_1 = np.vstack([
            0*np.identity(half_com_state_size),
            1*np.identity(half_com_state_size)])

        ## Eq(4)
        C_2 = np.hstack([np.identity(2), np.zeros((2,2))]) # C in Eq(2)
        D = -z_com / g * np.identity(zmp_state_size)
        Q = 0.1 * np.identity(zmp_state_size)

        ## Eq(6)
        # y.T*Q*y
        # = (C*x+D*u)*Q*(C*x+D*u).T
        # = x.T*C.T*Q*C*x + u.T*D.T*Q*D*u + x.T*C.T*Q*D*u + u.T*D.T*Q*C*X
        # = ..                            + 2*x.T*C.T*Q*D*u
        K, S = LinearQuadraticRegulator(A, B_1, C_2.T.dot(Q).dot(C_2), D.T.dot(Q).dot(D), C_2.T.dot(Q).dot(D))
        def V(x, u): # Assume constant z_com, we don't need tvLQR
            y = C_2.dot(x) + D.dot(u)
            def dJ_dx(x):
                return x.T.dot(S.T+S) # https://math.stackexchange.com/questions/20694/vector-derivative-w-r-t-its-transpose-fracdaxdxt
            x_d = A.dot(x) + B_1.dot(u)
            return y.T.dot(Q).dot(y) + dJ_dx(x).dot(x_d)
        self.V = V

    def create_qp1(self, plant_context):
        # Determine contact points
        lfoot_full_contact_pos = self.plant.CalcPointsPositions(plant_context, self.plant.GetFrameByName("l_foot"),
                lfoot_full_contact_points, self.plant.world_frame())
        lfoot_contact_points = lfoot_full_contact_points[:, np.where(lfoot_full_contact_pos[2,:] <= 0.0)[0]]
        rfoot_full_contact_pos = self.plant.CalcPointsPositions(plant_context, self.plant.GetFrameByName("r_foot"),
                rfoot_full_contact_points, self.plant.world_frame())
        rfoot_contact_points = rfoot_full_contact_points[:, np.where(rfoot_full_contact_pos[2,:] <= 0.0)[0]]
        print("lfoot contact z pos: " + str(lfoot_full_contact_pos[2]))
        print("lfoot # contacts: " + str(lfoot_contact_points.shape[1]))
        print("rfoot contact z pos: " + str(rfoot_full_contact_pos[2]))
        print("rfoot # contacts: " + str(rfoot_contact_points.shape[1]))
        N_c_lfoot = lfoot_contact_points.shape[1] # Num contacts per foot
        N_c_rfoot = rfoot_contact_points.shape[1] # Num contacts per foot
        N_c = N_c_lfoot + N_c_rfoot # num contact points

        ## Eq(7)
        H = self.plant.CalcMassMatrixViaInverseDynamics(plant_context)
        # Note that CalcGravityGeneralizedForces assumes the form Mv̇ + C(q, v)v = tau_g(q) + tau_app
        # while Eq(7) assumes gravity is accounted in C (on the left hand side)
        C_7 = self.plant.CalcBiasTerm(plant_context) - self.plant.CalcGravityGeneralizedForces(plant_context)
        B_7 = self.plant.MakeActuationMatrix()

        Phi_lfoot = self.plant.CalcJacobianTranslationalVelocity(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("l_foot"),
                lfoot_contact_points, self.plant.world_frame(), self.plant.world_frame())

        Phi_rfoot = self.plant.CalcJacobianTranslationalVelocity(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("r_foot"),
                rfoot_contact_points, self.plant.world_frame(), self.plant.world_frame())
        Phi = np.vstack([Phi_lfoot, Phi_rfoot])

        ## Eq(8)
        # From np.sort(np.nonzero(B_7)[0]) we know that indices 0-5 are the unactuated 6 DOF floating base and 6-35 are the actuated 30 DOF robot joints
        H_f = H[0:v_idx_act,:]
        H_a = H[v_idx_act:,:]
        C_f = C_7[0:v_idx_act]
        C_a = C_7[v_idx_act:]
        B_a = B_7[v_idx_act:,:]
        Phi_f_T = Phi.T[0:v_idx_act:,:]
        Phi_a_T = Phi.T[v_idx_act:,:]

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
        q_dd = prog.NewContinuousVariables(self.plant.num_velocities(), name="q_dd") # To ignore 6 DOF floating base
        self.q_dd = q_dd
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
        assert(J.shape == (N_c*N_f, TOTAL_DOF))

        eta = prog.NewContinuousVariables(J.shape[0], name="eta")
        self.eta = eta

        x = prog.NewContinuousVariables(com_state_size, name="x") # x_com, y_com, x_com_d, y_com_d
        self.x = x
        u = prog.NewContinuousVariables(half_com_state_size, name="u") # x_com_dd, y_com_dd
        self.u = u

        ## Eq(10)
        w = 10.0
        epsilon = 1.0e-8
        K_p = 0.8
        K_d = 0.0

        # For generalized positions, first 7 values are 4 quaternion + 3 x,y,z
        q = self.plant.GetPositions(plant_context)
        # For generalized velocities, first 6 values are 3 rotational velocities + 3 xd, yd, zd
        # Hence this not strictly the derivative of q
        q_d = self.plant.GetVelocities(plant_context)

        # Convert q, q_des to generalized velocities form
        q_err = calcPoseError(self.q_des, q)
        ignored_pose_indices = {3, 4} # Ignore x position, y position
        relevant_pose_indices = list(set(range(TOTAL_DOF)) - set(ignored_pose_indices))
        self.relevant_pose_indices = relevant_pose_indices
        q_dd_des = K_p*q_err - K_d*q_d
        self.q_dd_des = q_dd_des
        q_dd_err = q_dd_des[relevant_pose_indices] - q_dd[relevant_pose_indices]
        prog.AddCost(
                self.V(x, u)
                + w*((q_dd_err).dot(q_dd_err))
                + epsilon * np.sum(np.square(beta))
                + eta.dot(eta))

        ## Eq(11)
        eq11_lhs = H_f.dot(q_dd)+C_f
        eq11_rhs = Phi_f_T.dot(lambd)
        for i in range(eq11_lhs.size):
            prog.AddConstraint(eq11_lhs[i] == eq11_rhs[i])

        ## Eq(12)
        alpha = 1.0
        Jd_qd_lfoot = self.plant.CalcBiasTranslationalAcceleration(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("l_foot"),
                lfoot_contact_points, self.plant.world_frame(), self.plant.world_frame())
        Jd_qd_rfoot = self.plant.CalcBiasTranslationalAcceleration(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("r_foot"),
                rfoot_contact_points, self.plant.world_frame(), self.plant.world_frame())
        Jd_qd = np.concatenate([Jd_qd_lfoot.flatten(), Jd_qd_rfoot.flatten()])
        assert(Jd_qd.shape == (N_c*3,))
        eq12_lhs = J.dot(q_dd) + Jd_qd
        eq12_rhs = -alpha*J.dot(q_d) + eta
        for i in range(eq12_lhs.shape[0]):
            prog.AddConstraint(eq12_lhs[i] == eq12_rhs[i])

        ## Eq(13)
        def tau(q_dd, lambd):
            return np.linalg.inv(B_a).dot(H_a.dot(q_dd) + C_a - Phi_a_T.dot(lambd))
        self.tau = tau
        eq13_lhs = self.tau(q_dd, lambd)
        for i in range(eq13_lhs.shape[0]):
            prog.AddConstraint(eq13_lhs[i] >= tau_min)
            prog.AddConstraint(eq13_lhs[i] <= tau_max)

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

        ## Enforce x as com
        com = self.plant.CalcCenterOfMassPosition(plant_context)
        com_d = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                plant_context, JacobianWrtVariable.kV,
                self.plant.world_frame(), self.plant.world_frame()).dot(q_d)
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
                .dot(q_dd))
        prog.AddConstraint(u[0] == com_dd[0])
        prog.AddConstraint(u[1] == com_dd[1])

        ## Use PD to control z_com
        z_K_p = 0.5
        z_K_d = 0.0
        z_com_dd_des = -z_K_p*(com[2] - z_com) - z_K_d*(com_d[2])
        # print(f"z_com_dd_des = {z_com_dd_des}")
        if z_com_dd_des <= -g:
            prog.AddConstraint(com_dd[2] == -g)
        else:
            prog.AddConstraint(com_dd[2] == z_com_dd_des)

        return prog

    def calcTorqueOutput(self, context, output):
        if not self.start_time:
            self.start_time = context.get_time()

        ## FIXME: Start controller only after foot makes contact with ground
        if context.get_time() - self.start_time < 0.01:
            output.SetFromVector(np.zeros(30))
            return

        q_v = self.EvalVectorInput(context, self.input_q_v_idx).get_value()
        current_plant_context = self.plant.CreateDefaultContext()
        self.plant.SetPositionsAndVelocities(current_plant_context, q_v)

        prog = self.create_qp1(current_plant_context)
        result = Solve(prog)
        print(f"Success: {result.is_success()}, {result.get_optimal_cost()}")
        if not result.is_success():
            exit(-1)
        q_dd_sol = result.GetSolution(self.q_dd)
        lambd_sol = result.GetSolution(self.lambd)
        x_sol = result.GetSolution(self.x)
        u_sol = result.GetSolution(self.u)
        beta_sol = result.GetSolution(self.beta)
        eta_sol = result.GetSolution(self.eta)
        # print(f"V(x,y) = {self.V(x_sol, u_sol)}")
        # print(f"beta = {beta_sol}")
        # print(f"lambda = {lambd_sol}")
        tau = self.tau(q_dd_sol, lambd_sol)
        # print(f"tau = {tau}")
        output.SetFromVector(tau)

    def printCOMs(self, current_plant_context, result):
        q_dd_sol = result.GetSolution(self.q_dd)
        q_d = self.plant.GetVelocities(current_plant_context)
        com = self.plant.CalcCenterOfMassPosition(current_plant_context)
        com_d = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                current_plant_context, JacobianWrtVariable.kV,
                self.plant.world_frame(), self.plant.world_frame()).dot(q_d)
        com_dd = (
                self.plant.CalcBiasCenterOfMassTranslationalAcceleration(
                    current_plant_context, JacobianWrtVariable.kV,
                    self.plant.world_frame(), self.plant.world_frame())
                + self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                    current_plant_context, JacobianWrtVariable.kV,
                    self.plant.world_frame(), self.plant.world_frame())
                .dot(q_dd_sol))
        print(f"com = {com}")
        print(f"com_d = {com_d}")
        print(f"com_dd = {com_dd}")

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
    simulator.set_target_realtime_rate(0.5)
    simulator.AdvanceTo(5.0)

if __name__ == "__main__":
    main()
