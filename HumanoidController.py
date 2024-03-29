#!/usr/bin/python3

'''
This implements the paper
An Efficiently Solvable Quadratic Program for Stabilizing Dynamic Locomotion
by Scott Kuindersma, Frank Permenter, and Russ Tedrake

Other references:
[1]: Optimization-based Locomotion Planning, Estimation, and Control Design for the Atlas Humanoid Robot
by Scott Kuindersma, Robin Deits, Maurice Fallon, Andrés Valenzuela, Hongkai Dai, Frank Permenter, Twan Koolen, Pat Marion, Russ Tedrake

TODO:
- Convert to time-varying y_desired and z_com
- Proper use of Context to improve performance
'''

from Atlas import getJointLimitsSortedByActuator, getActuatorIndex, getJointValues, getJointIndexInGeneralizedVelocities
from Atlas import Atlas
import numpy as np
import random
from pydrake.all import (
        DiagramBuilder, MultibodyPlant, AddMultibodyPlantSceneGraph, SceneGraph, LeafSystem,
        MathematicalProgram, Solve, eq, le, ge, SnoptSolver,
        LinearQuadraticRegulator,
        JacobianWrtVariable,
        BasicVector, Quaternion, RigidTransform,
        ExternallyAppliedSpatialForce, SpatialForce, Value, List,
        Simulator, DrakeVisualizer, ConnectMeshcatVisualizer, MeshcatContactVisualizer
)
from functools import partial

import time
import pdb

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess

z_com = 1.220 # COM after 0.05s
zmp_state_size = 2
mbp_time_step = 1.0e-3

mu = 1.0 # Coefficient of friction
eta_min = -0.2
eta_max = 0.2

N_d = 4 # friction cone approximated as a i-pyramid
N_f = 3 # contact force dimension

g = -9.81

def normalize(q):
    return q / np.linalg.norm(q)

class HumanoidController(LeafSystem):
    '''
    is_wbc : bool
        toggles between COM/COP stabilization and whole body stabilization
        by changing the formulation of V
    '''
    def __init__(self, robot_ctor, is_wbc=False):
        LeafSystem.__init__(self)
        self.plant = MultibodyPlant(mbp_time_step)
        self.robot = robot_ctor(self.plant)
        self.contacts_per_frame = self.robot.CONTACTS_PER_FRAME
        self.is_wbc = is_wbc
        self.context = self.plant.CreateDefaultContext()
        self.plant.SetFreeBodyPose(self.context, self.plant.GetBodyByName("pelvis"), RigidTransform([0, 0, 0.93845]))
        self.q_nom = self.plant.GetPositions(self.context) # Nominal upright pose
        self.input_q_v_idx = self.DeclareVectorInputPort("q_v",
                BasicVector(self.plant.num_positions() + self.plant.num_velocities())).get_index()
        self.output_tau_idx = self.DeclareVectorOutputPort("tau", BasicVector(self.robot.NUM_ACTUATED_DOF), self.calcTorqueOutput).get_index()

        if self.is_wbc:
            com_dim = 3
            self.x_size = 4
            self.u_size = 2
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
            D = -z_com / g * np.identity(zmp_state_size)
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
            active_contacts_per_frame[frame] = contacts[:, np.where(contact_positions[2,:] <= 1e-4)[0]]

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
                J_foot = np.zeros((N_f*num_active_contacts, self.robot.nv))
                # TODO: Can this be simplified?
                for i in range(num_active_contacts):
                    J_foot[N_f*i:N_f*(i+1),:] = self.plant.CalcJacobianSpatialVelocity(
                        plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName(frame),
                        active_contacts[:,i], self.plant.world_frame(), self.plant.world_frame())[3:]
                J_foots.append(J_foot)
        J = np.vstack(J_foots)
        assert(J.shape == (N_c*N_f, self.robot.nv))

        eta = prog.NewContinuousVariables(J.shape[0], name="eta")
        self.eta = eta

        q = self.plant.GetPositions(plant_context)
        qd = self.plant.GetVelocities(plant_context)

        com = self.plant.CalcCenterOfMassPositionInWorld(plant_context)
        com_d = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                plant_context, JacobianWrtVariable.kV,
                self.plant.world_frame(), self.plant.world_frame()).dot(qd)

        x = np.array([com[0], com[1], com_d[0], com_d[1]])
        self.x = x
        u = prog.NewContinuousVariables(self.u_size, name="u") # x_com_dd, y_com_dd
        self.u = u

        ''' Eq(10) '''
        w = 0.01
        epsilon = 1.0e-8
        K_p = 10.0
        K_d = 4.0
        frame_weights = np.ones((self.robot.nv))

        # Convert q, q_nom to generalized velocities form
        q_err = self.plant.MapQDotToVelocity(plant_context, q_des - q)
        # print(f"Pelvis error: {q_err[0:3]}")
        ## FIXME: Not sure if it's a good idea to ignore the x, y, z position of pelvis
        # ignored_pose_indices = {3, 4, 5} # Ignore x position, y position
        ignored_pose_indices = {} # Ignore x position, y position
        relevant_pose_indices = list(set(range(self.robot.nv)) - set(ignored_pose_indices))
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
        prog.AddLinearConstraint(eq(eq11_lhs, eq11_rhs)).evaluator().set_description("Eq(11)")

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
        prog.AddLinearConstraint(eq(eq12_lhs, eq12_rhs)).evaluator().set_description("Eq(12)")

        ''' Eq(13) - 0.015s '''
        def tau(qdd, lambd):
            return self.B_a_inv.dot(H_a.dot(qdd) + C_a - Phi_a_T.dot(lambd))
        self.tau = tau
        eq13_lhs = self.tau(qdd, lambd)
        # AddBoundingBoxConstraint cannot be used with Expression eq13_lhs
        # prog.AddBoundingBoxConstraint(-self.sorted_max_efforts, self.sorted_max_efforts, eq13_lhs)
        prog.AddLinearConstraint(ge(eq13_lhs, -self.sorted_max_efforts)).evaluator().set_description("Eq(13 lower)")
        prog.AddLinearConstraint(le(eq13_lhs, self.sorted_max_efforts)).evaluator().set_description("Eq(13 upper)")

        ''' Eq(14) '''
        for j in range(N_c):
            beta_v = beta[:,j].dot(v[:,j])
            prog.AddLinearConstraint(eq(lambd[N_f*j:N_f*j+3], beta_v)).evaluator().set_description("Eq(14)")

        ''' Eq(15) '''
        prog.AddBoundingBoxConstraint(0, np.inf, beta).evaluator().set_description("Eq(15)")

        ''' Eq(16) '''
        prog.AddBoundingBoxConstraint(eta_min, eta_max, eta).evaluator().set_description("Eq(16)")

        ''' Enforce u as com_dd '''
        com_dd = (
                self.plant.CalcBiasCenterOfMassTranslationalAcceleration(
                    plant_context, JacobianWrtVariable.kV,
                    self.plant.world_frame(), self.plant.world_frame())
                + self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
                    plant_context, JacobianWrtVariable.kV,
                    self.plant.world_frame(), self.plant.world_frame())
                .dot(qdd))
        prog.AddLinearConstraint(u[0] == com_dd[0]).evaluator().set_description("u[0] == com_dd[0]")
        prog.AddLinearConstraint(u[1] == com_dd[1]).evaluator().set_description("u[1] == com_dd[1]")

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
                print(f"Joint {name} max reached")
                prog.AddLinearConstraint(qdd[q_idx] <= 0.0).evaluator().set_description(f"Joint[{q_idx}] upper limit")
            elif joint_pos <= limit.lower:
                print(f"Joint {name} min reached")
                prog.AddLinearConstraint(qdd[q_idx] >= 0.0).evaluator().set_description(f"Joint[{q_idx}] lower limit")

        return prog

    def calcTorqueOutput(self, context, output):
        print(f"===== TIME: {context.get_time()} =====")
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

        if not np.array_equal(q_v, self.plant.GetPositionsAndVelocities(self.context)):
            self.plant.SetPositionsAndVelocities(self.context, q_v)

        start_formulate_time = time.time()
        prog = self.create_qp1(self.context, V, q_des, v_des, vd_des)
        if not prog:
            print("Invalid program!")
            output.SetFromVector([0]*self.plant.num_actuated_dofs())
            return
        print(f"Formulate time: {time.time() - start_formulate_time}s")
        start_solve_time = time.time()
        result = Solve(prog)
        print(f"Solve time: {time.time() - start_solve_time}s")
        if not result.is_success():
            print(f"FAILED")
            output.SetFromVector([0]*self.plant.num_actuated_dofs())
            pdb.set_trace()
        print(f"Cost: {result.get_optimal_cost()}")
        qdd_sol = result.GetSolution(self.qdd)
        lambd_sol = result.GetSolution(self.lambd)
        x_sol = result.GetSolution(self.x)
        u_sol = result.GetSolution(self.u)
        beta_sol = result.GetSolution(self.beta)
        eta_sol = result.GetSolution(self.eta)

        tau = self.tau(qdd_sol, lambd_sol)
        output.SetFromVector(tau)

    def print_debug(self, context, result):
        com, comd, comdd = self.calcCOM(context, result)
        print(f"comdd z: {comdd[2]}")
        self.plant.EvalBodyPoseInWorld(context, self.plant.GetBodyByName("pelvis")).rotation().ToQuaternion().xyz()
        print(f"pelvis angular position = {Quaternion(normalize(q_v[0:4])).xyz()}")
        print(f"pelvis angular velocity = {q_v[Atlas.FLOATING_BASE_QUAT_DOF + Atlas.NUM_ACTUATED_DOF:FLOATING_BASE_QUAT_DOF + Atlas.NUM_ACTUATED_DOF + 3]}")
        print(f"pelvis angular acceleration = {qdd_sol[0:3]}")
        print(f"pelvis translational position = {q_v[4:7]}")
        print(f"pelvis translational velocity = {q_v[Atlas.FLOATING_BASE_QUAT_DOF + Atlas.NUM_ACTUATED_DOF + 3 : FLOATING_BASE_QUAT_DOF + Atlas.NUM_ACTUATED_DOF + 6]}")
        print(f"pelvis translational acceleration = {qdd_sol[3:6]}")
        print(f"beta = {beta_sol}")
        print(f"lambda z = {lambd_sol[2::3]}")
        print(f"Total force z = {np.sum(lambd_sol[2::3])}")

        interested_joints = [
                "back_bky",
                "l_leg_hpy",
                "r_leg_hpy"
        ]
        print(f"tau = {tau[getActuatorIndices(self.plant, interested_joints)]}")
        print(f"joint angles = {getJointValues(self.plant, interested_joints, context)}")
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

def rand_float(start, end):
    return (np.random.rand() * (end - start)) + start

def rand_2d_vec(mag):
    angle = rand_float(-np.pi, np.pi)
    return [mag*np.cos(angle), mag*np.sin(angle)]

class ForceDisturber(LeafSystem):
    def __init__(self, target_body_index, start_time, disturb_duration, disturb_period, magnitude):
        LeafSystem.__init__(self)
        self.target_body_index = target_body_index
        self.disturb_duration = disturb_duration
        self.disturb_period = disturb_period
        self.magnitude = magnitude
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
            if self.last_disturb_time is None or curr_time - self.last_disturb_time > self.disturb_period:
                self.last_disturb_time = curr_time
                self.start_disturb_time = curr_time
                self.last_disturb_time = curr_time
                self.force = rand_2d_vec(self.magnitude) + [0.0]
                self.position = [0.0, rand_float(-0.2, 0.2), rand_float(0.0, 0.5)]
                print(f"{curr_time}: Disturbing with {self.force} at {self.position}")
                # self.force = [5.0, 50.0, 0.0]

            if self.start_disturb_time is not None and curr_time - self.start_disturb_time > self.disturb_duration:
                self.start_disturb_time = None
                self.force = [0.0, 0.0, 0.0]
                print(f"Stop disturbing")

            test_force.p_BoBq_B = self.position
            test_force.F_Bq_W = SpatialForce(
                tau=[0., 0., 0.], f=self.force)
            spatial_forces_vector.set_value([test_force])

        else:
            spatial_forces_vector.set_value([])

def main():
    builder = DiagramBuilder()
    sim_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(mbp_time_step))
    sim_robot = Atlas(sim_plant, add_ground=True)
    sim_plant_context = sim_plant.CreateDefaultContext()

    controller = builder.AddSystem(HumanoidController(partial(Atlas, add_ground=False, simplified=False), is_wbc=False))
    controller.set_name("HumanoidController")

    disturber = builder.AddSystem(ForceDisturber(
        sim_plant.GetBodyByName("utorso").index(),
        start_time=4,
        disturb_duration=0.1,
        disturb_period=2,
        magnitude=120))
    builder.Connect(disturber.get_output_port(0), sim_plant.get_applied_spatial_force_input_port())

    builder.Connect(sim_plant.get_state_output_port(), controller.GetInputPort("q_v"))
    builder.Connect(controller.GetOutputPort("tau"), sim_plant.get_actuation_input_port())

    # Do not use drake-visualizer until I figure out how to load meshes from custom directory
    # DrakeVisualizer.AddToBuilder(builder=builder, scene_graph=scene_graph)

    proc, zmq_url, web_url = start_zmq_server_as_subprocess()
    visualizer = ConnectMeshcatVisualizer(builder=builder, scene_graph=scene_graph, zmq_url=zmq_url)

    contact_vis = builder.AddSystem(MeshcatContactVisualizer(
        meshcat_viz=visualizer,
        plant=sim_plant,
        contact_force_scale=400))
    contact_input_port = contact_vis.GetInputPort("contact_results")
    builder.Connect(
        sim_plant.GetOutputPort("contact_results"),
        contact_input_port)

    diagram = builder.Build()
    visualizer.load()
    visualizer.start_recording()
    diagram_context = diagram.CreateDefaultContext()
    sim_plant_context = diagram.GetMutableSubsystemContext(sim_plant, diagram_context)
    sim_plant.SetFreeBodyPose(sim_plant_context, sim_plant.GetBodyByName("pelvis"), RigidTransform([0, 0, 0.93845]))
    controller_context = diagram.GetMutableSubsystemContext(controller, diagram_context)
    controller.GetInputPort("y_des").FixValue(controller_context, np.array([0.1, 0.0]))

    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(0.1)
    simulator.AdvanceTo(20.0)
    visualizer.stop_recording()
    visualizer.publish_recording()

if __name__ == "__main__":
    main()
