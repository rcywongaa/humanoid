#!/usr/bin/python3

'''
This implements the paper
An Efficiently Solvable Quadratic Program for Stabilizing Dynamic Locomotion
by Scott Kuindersma, Frank Permenter, and Russ Tedrake

Other references:
[1]: Optimization-based Locomotion Planning, Estimation, and Control Design for the Atlas Humanoid Robot
by Scott Kuindersma, Robin Deits, Maurice Fallon, Andrés Valenzuela, Hongkai Dai, Frank Permenter, Twan Koolen, Pat Marion, Russ Tedrake
TODO:
Convert to time-varying y_desired and z_com
'''

from Atlas import load_atlas, set_atlas_initial_pose
from Atlas import getJointLimitsSortedByActuator, getActuatorIndex, getJointValues, getJointIndexInGeneralizedVelocities
from Atlas import Atlas
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
from pydrake.all import eq, le, ge
from pydrake.all import ExternallyAppliedSpatialForce, SpatialForce, Value, List

import time
import pdb

z_com = 1.220 # COM after 0.05s
zmp_state_size = 2
mbp_time_step = 1.0e-3

mu = 1.0 # Coefficient of friction, same as in load_atlas.py
eta_min = -0.2
eta_max = 0.2

N_d = 4 # friction cone approximated as a i-pyramid
N_f = 3 # contact force dimension

def normalize(q):
    return q / np.linalg.norm(q)

class HumanoidController(LeafSystem):
    '''
    is_wbc : bool
        toggles between COM/COP stabilization and whole body stabilization
        by changing the formulation of V
    '''
    def __init__(self, plant, contacts_per_frame, is_wbc=False):
        LeafSystem.__init__(self)
        self.plant = plant
        self.contacts_per_frame = contacts_per_frame
        self.is_wbc = is_wbc
        self.upright_context = self.plant.CreateDefaultContext()
        self.q_nom = self.plant.GetPositions(self.upright_context) # Nominal upright pose
        self.input_q_v_idx = self.DeclareVectorInputPort("q_v",
                BasicVector(self.plant.num_positions() + self.plant.num_velocities())).get_index()
        self.output_tau_idx = self.DeclareVectorOutputPort("tau", BasicVector(Atlas.NUM_ACTUATED_DOF), self.calcTorqueOutput).get_index()

        if self.is_wbc:
            com_dim = 3
            self.x_size = 2*com_dim
            self.u_size = com_dim
            self.input_r_des_idx = self.DeclareVectorInputPort("r_des", BasicVector(com_dim)).get_index()
            self.input_rd_des_idx = self.DeclareVectorInputPort("rd_des", BasicVector(com_dim)).get_index()
            self.input_rdd_des_idx = self.DeclareVectorInputPort("rdd_des", BasicVector(com_dim)).get_index()
            self.input_q_des_idx = self.DeclareVectorInputPort("q_des", BasicVector(self.plant.num_positions())).get_index()
            self.input_v_des_idx = self.DeclareVectorInputPort("v_des", BasicVector(self.plant.num_velocities())).get_index()
            self.input_vd_des_idx = self.DeclareVectorInputPort("vd_des", BasicVector(self.plant.num_velocities())).get_index()
            Q = 1.0 * np.identity(self.x_size)
            R = 0.1 * np.identity(self.u_size)
            A = np.vstack([
                np.hstack([0*np.identity(com_dim), 1*np.identity(com_dim)]),
                np.hstack([0*np.identity(com_dim), 0*np.identity(com_dim)])])
            B_1 = np.vstack([
                0*np.identity(com_dim),
                1*np.identity(com_dim)])
            K, S = LinearQuadraticRegulator(A, B_1, Q, R)
            def V_full(x, u, r, rd, rdd):
                x_bar = x - np.concatenate([r, rd])
                u_bar = u - rdd
                '''
                xd_bar = d(x - [r, rd].T)/dt
                        = xd - [rd, rdd].T
                        = Ax + Bu - [rd, rdd].T
                '''
                xd_bar = A.dot(x) + B_1.dot(u) - np.concatenate([rd, rdd])
                return x_bar.T.dot(Q).dot(x_bar) + u_bar.T.dot(R).dot(u_bar) + 2*x_bar.T.dot(S).dot(xd_bar)
            self.V_full = V_full

        else:
            # Only x, y coordinates of COM is considered
            com_dim = 2
            self.x_size = 2*com_dim
            self.u_size = com_dim
            self.input_y_des_idx = self.DeclareVectorInputPort("y_des", BasicVector(zmp_state_size)).get_index()
            ''' Eq(1) '''
            A = np.vstack([
                np.hstack([0*np.identity(com_dim), 1*np.identity(com_dim)]),
                np.hstack([0*np.identity(com_dim), 0*np.identity(com_dim)])])
            B_1 = np.vstack([
                0*np.identity(com_dim),
                1*np.identity(com_dim)])

            ''' Eq(4) '''
            C_2 = np.hstack([np.identity(2), np.zeros((2,2))]) # C in Eq(2)
            D = -z_com / Atlas.g * np.identity(zmp_state_size)
            Q = 1.0 * np.identity(zmp_state_size)

            ''' Eq(6) '''
            '''
            y.T*Q*y
            = (C*x+D*u)*Q*(C*x+D*u).T
            = x.T*C.T*Q*C*x + u.T*D.T*Q*D*u + x.T*C.T*Q*D*u + u.T*D.T*Q*C*X
            = ..                            + 2*x.T*C.T*Q*D*u
            '''
            K, S = LinearQuadraticRegulator(A, B_1, C_2.T.dot(Q).dot(C_2), D.T.dot(Q).dot(D), C_2.T.dot(Q).dot(D))
            # Use original formulation
            def V_full(x, u, y_des): # Assume constant z_com, we don't need tvLQR
                y = C_2.dot(x) + D.dot(u)
                def dJ_dx(x):
                    return x.T.dot(S.T+S) # https://math.stackexchange.com/questions/20694/vector-derivative-w-r-t-its-transpose-fracdaxdxt
                y_bar = y - y_des
                x_bar = x - np.concatenate([y_des, [0.0, 0.0]]) # FIXME: This doesn't seem right...
                # FIXME: xd_bar should depend on yd_des
                xd_bar = A.dot(x_bar) + B_1.dot(u)
                return y_bar.T.dot(Q).dot(y_bar) + dJ_dx(x_bar).dot(xd_bar)
            self.V_full = V_full

        # Calculate values that don't depend on context
        self.B_7 = self.plant.MakeActuationMatrix()
        # From np.sort(np.nonzero(B_7)[0]) we know that indices 0-5 are the unactuated 6 DOF floating base and 6-35 are the actuated 30 DOF robot joints
        self.v_idx_act = 6 # Start index of actuated joints in generalized velocities
        self.B_a = self.B_7[self.v_idx_act:,:]
        self.B_a_inv = np.linalg.inv(self.B_a)

        # Sort joint effort limits to be the same order as tau in Eq(13)
        self.sorted_max_efforts = np.array([entry[1].effort for entry in getJointLimitsSortedByActuator(self.plant)])

    def create_qp1(self, plant_context, V, q_des, v_des, vd_des):
        # Determine contact points
        contact_positions_per_frame = {}
        active_contacts_per_frame = {} # Note this should be in frame space
        for frame, contacts in self.contacts_per_frame.items():
            contact_positions = self.plant.CalcPointsPositions(
                    plant_context, self.plant.GetFrameByName(frame),
                    contacts, self.plant.world_frame())
            active_contacts_per_frame[frame] = contacts[:, np.where(contact_positions[2,:] <= 0.0)[0]]

        N_c = sum([active_contacts.shape[1] for active_contacts in active_contacts_per_frame.values()]) # num contact points
        if N_c == 0:
            print("Not in contact!")
            return None

        ''' Eq(7) '''
        H = self.plant.CalcMassMatrixViaInverseDynamics(plant_context)
        # Note that CalcGravityGeneralizedForces assumes the form Mv̇ + C(q, v)v = tau_g(q) + tau_app
        # while Eq(7) assumes gravity is accounted in C (on the left hand side)
        C_7 = self.plant.CalcBiasTerm(plant_context) - self.plant.CalcGravityGeneralizedForces(plant_context)
        B_7 = self.B_7

        # TODO: Double check
        Phi_foots = []
        for frame, active_contacts in active_contacts_per_frame.items():
            if active_contacts.size:
                Phi_foots.append(
                    self.plant.CalcJacobianTranslationalVelocity(
                        plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName(frame),
                        active_contacts, self.plant.world_frame(), self.plant.world_frame()))
        Phi = np.vstack(Phi_foots)

        ''' Eq(8) '''
        v_idx_act = self.v_idx_act
        H_f = H[0:v_idx_act,:]
        H_a = H[v_idx_act:,:]
        C_f = C_7[0:v_idx_act]
        C_a = C_7[v_idx_act:]
        B_a = self.B_a
        Phi_f_T = Phi.T[0:v_idx_act:,:]
        Phi_a_T = Phi.T[v_idx_act:,:]

        ''' Eq(9) '''
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

        ''' Quadratic Program I '''
        prog = MathematicalProgram()
        qdd = prog.NewContinuousVariables(self.plant.num_velocities(), name="qdd") # To ignore 6 DOF floating base
        self.qdd = qdd
        beta = prog.NewContinuousVariables(N_d,N_c, name="beta")
        self.beta = beta
        lambd = prog.NewContinuousVariables(N_f*N_c, name="lambda")
        self.lambd = lambd

        # Jacobians ignoring the 6DOF floating base
        J_foots = []
        for frame, active_contacts in active_contacts_per_frame.items():
            if active_contacts.size:
                num_active_contacts = active_contacts.shape[1]
                J_foot = np.zeros((N_f*num_active_contacts, Atlas.TOTAL_DOF))
                # TODO: Can this be simplified?
                for i in range(num_active_contacts):
                    J_foot[N_f*i:N_f*(i+1),:] = self.plant.CalcJacobianSpatialVelocity(
                        plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName(frame),
                        active_contacts[:,i], self.plant.world_frame(), self.plant.world_frame())[3:]
                J_foots.append(J_foot)
        J = np.vstack(J_foots)
        assert(J.shape == (N_c*N_f, Atlas.TOTAL_DOF))

        eta = prog.NewContinuousVariables(J.shape[0], name="eta")
        self.eta = eta

        x = prog.NewContinuousVariables(self.x_size, name="x") # x_com, y_com, x_com_d, y_com_d
        self.x = x
        u = prog.NewContinuousVariables(self.u_size, name="u") # x_com_dd, y_com_dd
        self.u = u

        ''' Eq(10) '''
        w = 0.01
        epsilon = 1.0e-8
        K_p = 10.0
        K_d = 2.0
        frame_weights = np.ones((Atlas.TOTAL_DOF))

        q = self.plant.GetPositions(plant_context)
        qd = self.plant.GetVelocities(plant_context)

        # Convert q, q_nom to generalized velocities form
        q_err = self.plant.MapQDotToVelocity(plant_context, q_des - q)
        print(f"Pelvis error: {q_err[0:3]}")
        ## FIXME: Not sure if it's a good idea to ignore the x, y, z position of pelvis
        # ignored_pose_indices = {3, 4, 5} # Ignore x position, y position
        ignored_pose_indices = {} # Ignore x position, y position
        relevant_pose_indices = list(set(range(Atlas.TOTAL_DOF)) - set(ignored_pose_indices))
        self.relevant_pose_indices = relevant_pose_indices
        qdd_ref = K_p*q_err + K_d*(v_des - qd) + vd_des # Eq(27) of [1]
        qdd_err = qdd_ref - qdd
        qdd_err = qdd_err*frame_weights
        qdd_err = qdd_err[relevant_pose_indices]
        prog.AddCost(
                V(x, u)
                + w*((qdd_err).dot(qdd_err))
                + epsilon * np.sum(np.square(beta))
                + eta.dot(eta))

        ''' Eq(11) - 0.003s '''
        eq11_lhs = H_f.dot(qdd)+C_f
        eq11_rhs = Phi_f_T.dot(lambd)
        prog.AddLinearConstraint(eq(eq11_lhs, eq11_rhs))

        ''' Eq(12) - 0.005s '''
        alpha = 0.1
        # TODO: Double check
        Jd_qd_foots = []
        for frame, active_contacts in active_contacts_per_frame.items():
            if active_contacts.size:
                Jd_qd_foot = self.plant.CalcBiasTranslationalAcceleration(
                    plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName(frame),
                    active_contacts, self.plant.world_frame(), self.plant.world_frame())
                Jd_qd_foots.append(Jd_qd_foot.flatten())
        Jd_qd = np.concatenate(Jd_qd_foots)
        assert(Jd_qd.shape == (N_c*3,))
        eq12_lhs = J.dot(qdd) + Jd_qd
        eq12_rhs = -alpha*J.dot(qd) + eta
        prog.AddLinearConstraint(eq(eq12_lhs, eq12_rhs))

        ''' Eq(13) - 0.015s '''
        def tau(qdd, lambd):
            return self.B_a_inv.dot(H_a.dot(qdd) + C_a - Phi_a_T.dot(lambd))
        self.tau = tau
        eq13_lhs = self.tau(qdd, lambd)
        prog.AddLinearConstraint(ge(eq13_lhs, -self.sorted_max_efforts))
        prog.AddLinearConstraint(le(eq13_lhs, self.sorted_max_efforts))

        ''' Eq(14) '''
        for j in range(N_c):
            beta_v = beta[:,j].dot(v[:,j])
            prog.AddLinearConstraint(eq(lambd[N_f*j:N_f*j+3], beta_v))

        ''' Eq(15) '''
        for b in beta.flat:
            prog.AddLinearConstraint(b >= 0.0)

        ''' Eq(16) '''
        prog.AddLinearConstraint(ge(eta, eta_min))
        prog.AddLinearConstraint(le(eta, eta_max))

        ''' Enforce x as com '''
        com = self.plant.CalcCenterOfMassPosition(plant_context)
        com_d = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                plant_context, JacobianWrtVariable.kV,
                self.plant.world_frame(), self.plant.world_frame()).dot(qd)
        prog.AddLinearConstraint(x[0] == com[0])
        prog.AddLinearConstraint(x[1] == com[1])
        prog.AddLinearConstraint(x[2] == com_d[0])
        prog.AddLinearConstraint(x[3] == com_d[1])

        ''' Enforce u as com_dd '''
        com_dd = (
                self.plant.CalcBiasCenterOfMassTranslationalAcceleration(
                    plant_context, JacobianWrtVariable.kV,
                    self.plant.world_frame(), self.plant.world_frame())
                + self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                    plant_context, JacobianWrtVariable.kV,
                    self.plant.world_frame(), self.plant.world_frame())
                .dot(qdd))
        prog.AddLinearConstraint(u[0] == com_dd[0])
        prog.AddLinearConstraint(u[1] == com_dd[1])

        ''' Respect joint limits '''
        for name, limit in Atlas.JOINT_LIMITS.items():
            # Get the corresponding joint value
            joint_pos = self.plant.GetJointByName(name).get_angle(plant_context)
            # Get the corresponding actuator index
            act_idx = getActuatorIndex(self.plant, name)
            # Use the actuator index to find the corresponding generalized coordinate index
            # q_idx = np.where(B_7[:,act_idx] == 1)[0][0]
            q_idx = getJointIndexInGeneralizedVelocities(self.plant, name)

            if joint_pos >= limit.upper:
                # print(f"Joint {name} max reached")
                prog.AddLinearConstraint(qdd[q_idx] <= 0.0)
            elif joint_pos <= limit.lower:
                # print(f"Joint {name} min reached")
                prog.AddLinearConstraint(qdd[q_idx] >= 0.0)

        return prog

    def calcTorqueOutput(self, context, output):
        q_v = self.EvalVectorInput(context, self.input_q_v_idx).get_value()
        if self.is_wbc:
            r = self.EvalVectorInput(context, self.input_r_des_idx).get_value()
            rd = self.EvalVectorInput(context, self.input_rd_des_idx).get_value()
            rdd = self.EvalVectorInput(context, self.input_rdd_des_idx).get_value()
            V = lambda x, u : self.V_full(x, u, r, rd, rdd)
            q_des = self.EvalVectorInput(context, self.input_q_des_idx).get_value()
            v_des = self.EvalVectorInput(context, self.input_v_des_idx).get_value()
            vd_des = self.EvalVectorInput(context, self.input_vd_des_idx).get_value()
        else:
            y_des = self.EvalVectorInput(context, self.input_y_des_idx).get_value()
            V = lambda x, u : self.V_full(x, u, y_des)
            q_des = self.q_nom
            v_des = [0.0] * self.plant.num_velocities()
            vd_des = [0.0] * self.plant.num_velocities()
        current_plant_context = self.plant.CreateDefaultContext()
        self.plant.SetPositionsAndVelocities(current_plant_context, q_v)

        start_formulate_time = time.time()
        prog = self.create_qp1(current_plant_context, V, q_des, v_des, vd_des)
        if not prog:
            print("Invalid program!")
            output.SetFromVector([0]*self.plant.num_actuated_dofs())
            return
        # print(f"Formulate time: {time.time() - start_formulate_time}s")
        start_solve_time = time.time()
        result = Solve(prog)
        # print(f"Solve time: {time.time() - start_solve_time}s")
        if not result.is_success():
            print(f"FAILED")
            output.SetFromVector([0]*self.plant.num_actuated_dofs())
        # print(f"Cost: {result.get_optimal_cost()}")
        qdd_sol = result.GetSolution(self.qdd)
        lambd_sol = result.GetSolution(self.lambd)
        x_sol = result.GetSolution(self.x)
        u_sol = result.GetSolution(self.u)
        beta_sol = result.GetSolution(self.beta)
        eta_sol = result.GetSolution(self.eta)

        tau = self.tau(qdd_sol, lambd_sol)
        output.SetFromVector(tau)

        # com, comd, comdd = self.calcCOM(current_plant_context, result)
        # print(f"comdd z: {comdd[2]}")
        # self.plant.EvalBodyPoseInWorld(current_plant_context, self.plant.GetBodyByName("pelvis")).rotation().ToQuaternion().xyz()
        # print(f"pelvis angular position = {Quaternion(normalize(q_v[0:4])).xyz()}")
        # print(f"pelvis angular velocity = {q_v[Atlas.FLOATING_BASE_QUAT_DOF + Atlas.NUM_ACTUATED_DOF:Atlas.FLOATING_BASE_QUAT_DOF + Atlas.NUM_ACTUATED_DOF + 3]}")
        # print(f"pelvis angular acceleration = {qdd_sol[0:3]}")
        # print(f"pelvis translational position = {q_v[4:7]}")
        # print(f"pelvis translational velocity = {q_v[Atlas.FLOATING_BASE_QUAT_DOF + Atlas.NUM_ACTUATED_DOF + 3 : Atlas.FLOATING_BASE_QUAT_DOF + Atlas.NUM_ACTUATED_DOF + 6]}")
        # print(f"pelvis translational acceleration = {qdd_sol[3:6]}")
        # print(f"beta = {beta_sol}")
        # print(f"lambda z = {lambd_sol[2::3]}")
        # print(f"Total force z = {np.sum(lambd_sol[2::3])}")

        # interested_joints = [
                # "back_bky",
                # "l_leg_hpy",
                # "r_leg_hpy"
        # ]
        # print(f"tau = {tau[getActuatorIndices(self.plant, interested_joints)]}")
        # print(f"joint angles = {getJointValues(self.plant, interested_joints, current_plant_context)}")
        # print("========================================")

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

def rand_float(start, end):
    return (np.random.rand() * (end - start)) + start

class ForceDisturber(LeafSystem):
    def __init__(self, target_body_index, start_time, disturb_duration, disturb_period):
        LeafSystem.__init__(self)
        self.target_body_index = target_body_index
        self.disturb_duration = disturb_duration
        self.disturb_period = disturb_period
        self.start_time = start_time
        forces_cls = Value[List[ExternallyAppliedSpatialForce]]
        self.DeclareAbstractOutputPort(
            "spatial_forces_vector",
            lambda: forces_cls(),
            self.DoCalcAbstractOutput)
        self.last_disturb_time = None
        self.start_disturb_time = None
        self.force = [0.0, 0.0, 0.0]

    def DoCalcAbstractOutput(self, context, spatial_forces_vector):
        curr_time = context.get_time()
        if curr_time > self.start_time:
            test_force = ExternallyAppliedSpatialForce()
            test_force.body_index = self.target_body_index
            test_force.p_BoBq_B = np.zeros(3)
            if self.last_disturb_time is None or curr_time - self.last_disturb_time > self.disturb_period:
                self.last_disturb_time = curr_time
                self.start_disturb_time = curr_time
                self.last_disturb_time = curr_time
                # self.force = [rand_float(-5, 5), rand_float(-10, 10), 0.0]
                self.force = [5.0, 20.0, 0.0]

            if self.start_disturb_time is not None and curr_time - self.start_disturb_time < self.disturb_duration:
                print(f"Disturbing with {self.force}")
                test_force.F_Bq_W = SpatialForce(
                    tau=[0., 0., 0.], f=self.force)
            else:
                self.start_disturb_time = None
                test_force.F_Bq_W = SpatialForce(
                    tau=[0., 0., 0.], f=[0., 0., 0.])

            spatial_forces_vector.set_value([test_force])

        else:
            spatial_forces_vector.set_value([])

def main():
    builder = DiagramBuilder()
    sim_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(mbp_time_step))
    load_atlas(sim_plant, add_ground=True)
    sim_plant_context = sim_plant.CreateDefaultContext()

    controller_plant = MultibodyPlant(mbp_time_step)
    load_atlas(controller_plant)
    controller = builder.AddSystem(HumanoidController(controller_plant, Atlas.CONTACTS_PER_FRAME, is_wbc=False))
    controller.set_name("HumanoidController")

    disturber = builder.AddSystem(ForceDisturber(
        sim_plant.GetBodyByName("utorso").index(), 2, 0.5, 4))
    builder.Connect(disturber.get_output_port(0), sim_plant.get_applied_spatial_force_input_port())

    builder.Connect(sim_plant.get_state_output_port(), controller.GetInputPort("q_v"))
    builder.Connect(controller.GetOutputPort("tau"), sim_plant.get_actuation_input_port())

    ConnectContactResultsToDrakeVisualizer(builder=builder, plant=sim_plant)
    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    sim_plant_context = diagram.GetMutableSubsystemContext(sim_plant, diagram_context)
    set_atlas_initial_pose(sim_plant, sim_plant_context)
    controller_context = diagram.GetMutableSubsystemContext(controller, diagram_context)
    controller.GetInputPort("y_des").FixValue(controller_context, np.array([0.1, 0.0]))

    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(0.1)
    simulator.AdvanceTo(10.0)

if __name__ == "__main__":
    main()
