#!/usr/bin/python3

# This implements the paper
# An Efficiently Solvable Quadratic Program for Stabilizing Dynamic Locomotion
# by Scott Kuindersma, Frank Permenter, and Russ Tedrake

# TODO:
# Convert plant_context to be dependent on the time varying state via plant output ports

from load_atlas import load_atlas
import numpy as np
from pydrake.systems.framework import DiagramBuilder
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.systems.controllers import LinearQuadraticRegulator
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.geometry import ConnectDrakeVisualizer, SceneGraph
from pydrake.systems.analysis import Simulator
import pdb

if __name__ == "__main__":
    com_state_size = 4
    half_com_state_size = int(com_state_size/2.0)
    zmp_state_size = 2

    ## Eq(1)
    A = np.vstack([
        np.hstack([0*np.identity(half_com_state_size), 1*np.identity(half_com_state_size)]),
        np.hstack([0*np.identity(half_com_state_size), 0*np.identity(half_com_state_size)])])
    B_1 = np.vstack([
        0*np.identity(half_com_state_size),
        1*np.identity(half_com_state_size)])

    ## Eq(4)
    C_2 = np.hstack([np.identity(2), np.zeros((2,2))]) # C in Eq(2)
    z_com = 0.5
    g = 9.81
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

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(1.0e-3))
    load_atlas(plant)
    plant_context = plant.CreateDefaultContext()
    upright_context = plant.CreateDefaultContext()

    ## Eq(7)
    H = plant.CalcMassMatrixViaInverseDynamics(plant_context)
    C_7 = plant.CalcBiasTerm(plant_context) # C in Eq(7)
    B_7 = plant.MakeActuationMatrix()

    # Assume forces are applied at the center of the foot for now
    l_foot_contact_points = np.array([[0.0, 0.0, 0.0]]).T
    Phi_lfoot = plant.CalcJacobianTranslationalVelocity(
            plant_context, JacobianWrtVariable.kV, plant.GetFrameByName("l_foot"),
            l_foot_contact_points, plant.world_frame(), plant.world_frame())
    r_foot_contact_points = np.array([[0.0, 0.0, 0.0]]).T
    Phi_rfoot = plant.CalcJacobianTranslationalVelocity(
            plant_context, JacobianWrtVariable.kV, plant.GetFrameByName("r_foot"),
            r_foot_contact_points, plant.world_frame(), plant.world_frame())
    Phi = np.vstack([Phi_lfoot, Phi_rfoot])

    ## Eq(8)
    # From np.sort(np.nonzero(B_7)[0]) we know that indices 0-5 are the unactuated 6 DOF floating base and 6-35 are the actuated 30 DOF robot joints
    v_idx_act = 6 # Start index of actuated joints in generalized velocities
    H_f = H[0:v_idx_act,:]
    H_a = H[v_idx_act:,:]
    C_f = C_7[0:v_idx_act]
    C_a = C_7[v_idx_act:]
    B_a = B_7[v_idx_act:,:]
    Phi_f_T = Phi.T[0:v_idx_act:,:]
    Phi_a_T = Phi.T[v_idx_act:,:]



    ## Quadratic Program I
    prog = MathematicalProgram()
    q_dd = prog.NewContinuousVariables(plant.num_velocities(), name="q_dd") # Ignore 6 DOF floating base
    N_c = 2 # num contact points
    N_d = 4 # friction cone approximated as a i-pyramid
    N_f = 3 # contact force dimension
    beta = prog.NewContinuousVariables(N_d,N_c, name="beta")
    lambd = prog.NewContinuousVariables(N_f*N_c, name="lambda")

    # Jacobians inoring the 6DOF floating base
    J_lfoot = plant.CalcJacobianSpatialVelocity(
            plant_context, JacobianWrtVariable.kV, plant.GetFrameByName("l_foot"),
            l_foot_contact_points, plant.world_frame(), plant.world_frame())
    J_rfoot = plant.CalcJacobianSpatialVelocity(
            plant_context, JacobianWrtVariable.kV, plant.GetFrameByName("r_foot"),
            r_foot_contact_points, plant.world_frame(), plant.world_frame())
    J = J_lfoot + J_rfoot

    eta = prog.NewContinuousVariables(J.shape[0], name="eta")

    w = 0.5
    epsilon = 1.0e-8

    x = prog.NewContinuousVariables(com_state_size, name="x") # x_com, y_com, x_com_d, y_com_d
    u = prog.NewContinuousVariables(half_com_state_size, name="u") # x_com_dd, y_com_dd

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
    v = np.zeros((N_d, N_c, 3))
    for i in range(N_d):
        for j in range(N_c):
            v[i,j] = (n+mu*d)[:,i]

    ## Eq(10)
    K_p = 0.2
    K_d = 0.1

    # For generalized positions, first 7 values are 3 x,y,z + 4 quaternion
    q_idx_act = 7 # Start index of actuated joints in generalized positions
    q_des = plant.GetPositions(upright_context)
    q = plant.GetPositions(plant_context)
    # For generalized velocities, first 6 values are 3 xd, yd, zd + 3 rotational velocities
    q_d = plant.GetVelocities(plant_context)

    q_dd_des = K_p*(q_des[q_idx_act:] - q[q_idx_act:]) - K_d*q_d[v_idx_act:]
    # We only care about the pose, not the floating base
    q_dd_err = q_dd_des - q_dd[v_idx_act:]
    prog.AddCost(
            V(x, u)
            + w*((q_dd_err).dot(q_dd_err))
            + epsilon * np.sum(np.square(beta))
            + eta.dot(eta))

    ## Eq(11)
    for i in range(H_f.shape[0]):
        prog.AddConstraint((H_f.dot(q_dd)+C_f)[i] == (Phi_f_T.dot(lambd))[0])

    ## Eq(12)
    alpha = 1.0
    Jd_qd_lfoot = plant.CalcBiasTranslationalAcceleration(
            plant_context, JacobianWrtVariable.kV, plant.GetFrameByName("l_foot"),
            l_foot_contact_points, plant.world_frame(), plant.world_frame())
    Jd_qd_rfoot = plant.CalcBiasTranslationalAcceleration(
            plant_context, JacobianWrtVariable.kV, plant.GetFrameByName("r_foot"),
            l_foot_contact_points, plant.world_frame(), plant.world_frame())
    Jd_qd = np.vstack([Jd_qd_lfoot, Jd_qd_rfoot]).flatten()
    eq12_lhs = J.dot(q_dd) + Jd_qd
    eq12_rhs = -alpha*J.dot(q_d) + eta
    for i in range(eq12_lhs.shape[0]):
        prog.AddConstraint(eq12_lhs[i] == eq12_rhs[i])

    ## Eq(13)
    eq13_lhs = np.linalg.inv(B_a).dot(H_a.dot(q_dd) + C_a - Phi_a_T.dot(lambd))
    tau_min = -10.0
    tau_max = 10.0
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



    result = Solve(prog)
    print(f"Success: {result.is_success()}")
    q_sol = result.GetSolution(q)
    pdb.set_trace()

    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
