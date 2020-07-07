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
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import BasicVector, LeafSystem

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

l_foot_contact_points = np.array([
    [-0.0876,0.066,-0.07645], # left heel
    [-0.0876,-0.0626,-0.07645], # right heel
    [0.1728,0.066,-0.07645], # left toe
    [0.1728,-0.0626,-0.07645], # right toe
    [0.086,0.066,-0.07645], # left midfoot_front
    [0.086,-0.0626,-0.07645], # right midfoot_front
    [-0.0008,0.066,-0.07645], # left midfoot_rear
    [-0.0008,-0.0626,-0.07645] # right midfoot_rear
]).T

r_foot_contact_points = np.array([
    [-0.0876,0.0626,-0.07645], # left heel
    [-0.0876,-0.066,-0.07645], # right heel
    [0.1728,0.0626,-0.07645], # left toe
    [0.1728,-0.066,-0.07645], # right toe
    [0.086,0.0626,-0.07645], # left midfoot_front
    [0.086,-0.066,-0.07645], # right midfoot_front
    [-0.0008,0.0626,-0.07645], # left midfoot_rear
    [-0.0008,-0.066,-0.07645] # right midfoot_rear
]).T
N_c_foot = l_foot_contact_points.shape[1] # Num contacts per foot
N_c = 2*N_c_foot # num contact points
N_d = 4 # friction cone approximated as a i-pyramid
N_f = 3 # contact force dimension

v_idx_act = 6 # Start index of actuated joints in generalized velocities

class HumanoidController(LeafSystem):
    def __init__(self):
        self.start_time = None
        LeafSystem.__init__(self)
        self.plant = MultibodyPlant(mbp_time_step)
        load_atlas(self.plant)
        self.upright_context = self.plant.CreateDefaultContext()
        set_atlas_initial_pose(self.plant, self.upright_context)

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
        ## Eq(7)
        H = self.plant.CalcMassMatrixViaInverseDynamics(plant_context)
        C_7 = self.plant.CalcBiasTerm(plant_context) # C in Eq(7)
        B_7 = self.plant.MakeActuationMatrix()

        Phi_lfoot = self.plant.CalcJacobianTranslationalVelocity(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("l_foot"),
                l_foot_contact_points, self.plant.world_frame(), self.plant.world_frame())

        Phi_rfoot = self.plant.CalcJacobianTranslationalVelocity(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("r_foot"),
                r_foot_contact_points, self.plant.world_frame(), self.plant.world_frame())
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

        ## Quadratic Program I
        prog = MathematicalProgram()
        q_dd = prog.NewContinuousVariables(self.plant.num_velocities(), name="q_dd") # Ignore 6 DOF floating base
        self.q_dd = q_dd
        beta = prog.NewContinuousVariables(N_d,N_c, name="beta")
        self.beta = beta
        lambd = prog.NewContinuousVariables(N_f*N_c, name="lambda")
        self.lambd = lambd

        # Jacobians ignoring the 6DOF floating base
        J_lfoot = np.zeros((FLOATING_BASE_DOF*N_c_foot, TOTAL_DOF))
        for i in range(N_c_foot):
            J_lfoot[FLOATING_BASE_DOF*i:FLOATING_BASE_DOF*(i+1),:] = self.plant.CalcJacobianSpatialVelocity(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("l_foot"),
                l_foot_contact_points[:,i], self.plant.world_frame(), self.plant.world_frame())
        J_rfoot = np.zeros((FLOATING_BASE_DOF*N_c_foot, TOTAL_DOF))
        for i in range(N_c_foot):
            J_rfoot[FLOATING_BASE_DOF*i:FLOATING_BASE_DOF*(i+1),:] = self.plant.CalcJacobianSpatialVelocity(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("r_foot"),
                r_foot_contact_points[:,i], self.plant.world_frame(), self.plant.world_frame())

        J = J_lfoot + J_rfoot
        assert(J.shape == (N_c*N_f, TOTAL_DOF))

        eta = prog.NewContinuousVariables(J.shape[0], name="eta")
        self.eta = eta

        self.x = prog.NewContinuousVariables(com_state_size, name="x") # x_com, y_com, x_com_d, y_com_d
        x = self.x
        self.u = prog.NewContinuousVariables(half_com_state_size, name="u") # x_com_dd, y_com_dd
        u = self.u

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
        mu = 0.2
        v = np.zeros((N_d, N_c, N_f))
        for i in range(N_d):
            for j in range(N_c):
                v[i,j] = (n+mu*d)[:,i]

        ## Eq(10)
        w = 10.0
        epsilon = 1.0e-8
        K_p = 0.8
        K_d = 0.0

        # For generalized positions, first 7 values are 3 x,y,z + 4 quaternion
        q_idx_act = 7 # Start index of actuated joints in generalized positions
        q_des = self.plant.GetPositions(self.upright_context)
        q = self.plant.GetPositions(plant_context)
        # For generalized velocities, first 6 values are 3 xd, yd, zd + 3 rotational velocities
        # Hence this not strictly the derivative of q
        q_d = self.plant.GetVelocities(plant_context)

        q_dd_des = K_p*(q_des[q_idx_act:] - q[q_idx_act:]) - K_d*q_d[v_idx_act:]
        self.q_dd_des = q_dd_des
        # We only care about the pose, not the floating base
        q_dd_err = q_dd_des - q_dd[v_idx_act:]
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
                l_foot_contact_points, self.plant.world_frame(), self.plant.world_frame())
        Jd_qd_rfoot = self.plant.CalcBiasTranslationalAcceleration(
                plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName("r_foot"),
                r_foot_contact_points, self.plant.world_frame(), self.plant.world_frame())
        Jd_qd = np.vstack([Jd_qd_lfoot, Jd_qd_rfoot]).flatten()
        eq12_lhs = J.dot(q_dd) + Jd_qd
        eq12_rhs = -alpha*J.dot(q_d) + eta
        for i in range(eq12_lhs.shape[0]):
            prog.AddConstraint(eq12_lhs[i] == eq12_rhs[i])

        ## Eq(13)
        def tau(q_dd, lambd):
            return np.linalg.inv(B_a).dot(H_a.dot(q_dd) + C_a - Phi_a_T.dot(lambd))
        self.tau = tau
        eq13_lhs = self.tau(q_dd, lambd)
        tau_min = -100.0
        tau_max = 100.0
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
        eta_min = -0.2*np.ones(J.shape[0])
        eta_max = 0.2*np.ones(J.shape[0])
        for i in range(eta.shape[0]):
            prog.AddConstraint(eta[i] >= eta_min[i])
            prog.AddConstraint(eta[i] <= eta_max[i])

        ## Enforce x as com
        com_position = self.plant.CalcCenterOfMassPosition(plant_context)
        com_position_d = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                plant_context, JacobianWrtVariable.kV,
                self.plant.world_frame(), self.plant.world_frame()).dot(q_d)
        prog.AddConstraint(x[0] == com_position[0])
        prog.AddConstraint(x[1] == com_position[1])
        prog.AddConstraint(x[2] == com_position_d[0])
        prog.AddConstraint(x[3] == com_position_d[1])

        ## Enforce u as com_dd
        com_position_dd = (
                self.plant.CalcBiasCenterOfMassTranslationalAcceleration(
                    plant_context, JacobianWrtVariable.kV,
                    self.plant.world_frame(), self.plant.world_frame())
                + self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                    plant_context, JacobianWrtVariable.kV,
                    self.plant.world_frame(), self.plant.world_frame())
                .dot(q_dd))
        prog.AddConstraint(u[0] == com_position_dd[0])
        prog.AddConstraint(u[1] == com_position_dd[1])

        ## Use PD to control z_com
        z_K_p = 0.5
        z_K_d = 0.2
        prog.AddConstraint(com_position_dd[2] == -z_K_p*(com_position[2] - z_com) - z_K_d*(com_position_d[2]))
        # prog.AddConstraint(com_position_dd[2] == -z_K_d*(com_position_d[2]))

        return prog

    def calcTorqueOutput(self, context, output):
        if not self.start_time:
            self.start_time = time.time()

        ## FIXME: Start controller only after foot makes contact with ground
        # if True:
        if time.time() - self.start_time < 0.2:
            output.SetFromVector(np.zeros(30))
            return
        q_v = self.EvalVectorInput(context, self.input_q_v_idx).get_value()
        # print(f"q = {q_v[0:37]}")
        current_plant_context = self.plant.CreateDefaultContext()
        self.plant.SetPositionsAndVelocities(current_plant_context, q_v)
        # Use this to obtain z_com
        # print(f"COM = {self.plant.CalcCenterOfMassPosition(current_plant_context)}")

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
        q_dd_err = self.q_dd_des - q_dd_sol[v_idx_act:]
        print(f"q_dd_err squared = {q_dd_err.dot(q_dd_err)}")
        # print(f"beta = {beta_sol}")
        # print(f"lambda = {lambd_sol}")
        tau = self.tau(q_dd_sol, lambd_sol)
        # print(f"tau = {tau}")
        output.SetFromVector(tau)

def main():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(mbp_time_step))
    load_atlas(plant, add_ground=True)
    plant_context = plant.CreateDefaultContext()

    controller = builder.AddSystem(HumanoidController())
    controller.set_name("HumanoidController")

    builder.Connect(plant.get_state_output_port(), controller.GetInputPort("q_v"))
    builder.Connect(controller.GetOutputPort("tau"), plant.get_actuation_input_port())

    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    set_atlas_initial_pose(plant, plant_context)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(0.2)
    simulator.AdvanceTo(4.0)

if __name__ == "__main__":
    main()
