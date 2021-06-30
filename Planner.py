'''
Adapted from http://underactuated.mit.edu/humanoids.html#example1

Implements paper:
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics
by Hongkai Dai, Andrés Valenzuela and Russ Tedrake
'''
from LittleDog import LittleDog
from Atlas import Atlas

import pdb
from functools import partial
import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, ConnectMeshcatVisualizer, RigidTransform, Simulator, PidController
from pydrake.all import (
    MultibodyPlant, JointIndex, RotationMatrix, PiecewisePolynomial, JacobianWrtVariable,
    MathematicalProgram, Solve, eq, AutoDiffXd, autoDiffToGradientMatrix, SnoptSolver,
    initializeAutoDiffGivenGradientMatrix, autoDiffToValueMatrix, autoDiffToGradientMatrix,
    AddUnitQuaternionConstraintOnPlant, PositionConstraint, OrientationConstraint
)
from pydrake.common.containers import namedview

import sys
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(
    server_args=['--ngrok_http_tunnel'] if 'google.colab' in sys.modules else [])

running_as_notebook = True

# Need this because a==b returns True even if a = AutoDiffXd(1, [1, 2]), b= AutoDiffXd(2, [3, 4])
# That's the behavior of AutoDiffXd in C++, also.
def autoDiffArrayEqual(a,b):
    return np.array_equal(a, b) and np.array_equal(autoDiffToGradientMatrix(a), autoDiffToGradientMatrix(b))

# TODO: promote this to drake (and make a version with model_instance)
def MakeNamedViewPositions(mbp, view_name):
    names = [None]*mbp.num_positions()
    for ind in range(mbp.num_joints()): 
        joint = mbp.get_joint(JointIndex(ind))
        # TODO: Handle planar joints, etc.
        if joint.num_positions() != 1:
            pdb.set_trace()
        assert(joint.num_positions() == 1)
        names[joint.position_start()] = joint.name()
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        start = body.floating_positions_start()
        body_name = body.name()
        names[start] = body_name+'_qw'
        names[start+1] = body_name+'_qx'
        names[start+2] = body_name+'_qy'
        names[start+3] = body_name+'_qz'
        names[start+4] = body_name+'_x'
        names[start+5] = body_name+'_y'
        names[start+6] = body_name+'_z'
    return namedview(view_name, names)

def MakeNamedViewVelocities(mbp, view_name):
    names = [None]*mbp.num_velocities()
    for ind in range(mbp.num_joints()): 
        joint = mbp.get_joint(JointIndex(ind))
        # TODO: Handle planar joints, etc.
        assert(joint.num_velocities() == 1)
        names[joint.velocity_start()] = joint.name()
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        start = body.floating_velocities_start() - mbp.num_positions()
        body_name = body.name()
        names[start] = body_name+'_wx'
        names[start+1] = body_name+'_wy'
        names[start+2] = body_name+'_wz'
        names[start+3] = body_name+'_vx'
        names[start+4] = body_name+'_vy'
        names[start+5] = body_name+'_vz'
    return namedview(view_name, names)

def gait_optimization(robot_ctor):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    robot = robot_ctor(plant)
    plant.Finalize()
    visualizer = ConnectMeshcatVisualizer(builder, 
        scene_graph=scene_graph, 
        zmq_url=zmq_url)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    robot.set_home(plant, plant_context)
    visualizer.load()
    diagram.Publish(context)

    q0 = plant.GetPositions(plant_context)
    body_frame = plant.GetFrameByName(robot.get_body_name())

    PositionView = MakeNamedViewPositions(plant, "Positions")
    VelocityView = MakeNamedViewVelocities(plant, "Velocities")

    mu = 1 # rubber on rubber
    total_mass = robot.get_total_mass(context)
    gravity = plant.gravity_field().gravity_vector()
    g = 9.81
    
    nq = 12
    contact_frame = robot.get_contact_frames()

    in_stance = robot.get_stance_schedule()
    N = robot.get_num_timesteps()
    is_laterally_symmetric = robot.get_laterally_symmetric()
    check_self_collision = robot.get_check_self_collision()
    stride_length = robot.get_stride_length()
    speed = robot.get_speed()
    
    T = stride_length / speed
    if is_laterally_symmetric:
        T = T / 2.0

    prog = MathematicalProgram()        

    # Time steps    
    h = prog.NewContinuousVariables(N-1, "h")
    prog.AddBoundingBoxConstraint(0.5*T/N, 2.0*T/N, h) 
    prog.AddLinearConstraint(sum(h) >= .9*T)
    prog.AddLinearConstraint(sum(h) <= 1.1*T)

    # Create one context per timestep (to maximize cache hits)
    context = [plant.CreateDefaultContext() for i in range(N)]
    # We could get rid of this by implementing a few more Jacobians in MultibodyPlant:
    ad_plant = plant.ToAutoDiffXd()

    # Joint positions and velocities
    nq = plant.num_positions()
    nv = plant.num_velocities()
    q = prog.NewContinuousVariables(nq, N, "q")
    v = prog.NewContinuousVariables(nv, N, "v")
    q_view = PositionView(q)
    v_view = VelocityView(v)
    q0_view = PositionView(q0)
    # Joint costs
    q_cost = PositionView([1]*nq)
    v_cost = VelocityView([1]*nv)
    q_cost.body_x = 0
    q_cost.body_y = 0
    q_cost.body_qx = 0
    q_cost.body_qy = 0
    q_cost.body_qz = 0
    q_cost.body_qw = 0
    q_cost.front_left_hip_roll = 5
    q_cost.front_right_hip_roll = 5
    q_cost.back_left_hip_roll = 5
    q_cost.back_right_hip_roll = 5
    v_cost.body_vx = 0
    v_cost.body_wx = 0
    v_cost.body_wy = 0
    v_cost.body_wz = 0
    for n in range(N):
        # Joint limits
        prog.AddBoundingBoxConstraint(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits(), q[:,n])
        # Joint velocity limits
        prog.AddBoundingBoxConstraint(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits(), v[:,n])
        # Unit quaternions
        AddUnitQuaternionConstraintOnPlant(plant, q[:,n], prog)
        # Body orientation
        prog.AddConstraint(OrientationConstraint(plant, 
                                                 body_frame, RotationMatrix(),
                                                 plant.world_frame(), RotationMatrix(), 
                                                 0.1, context[n]), q[:,n])
        # Initial guess for all joint angles is the home position
        prog.SetInitialGuess(q[:,n], q0)  # Solvers get stuck if the quaternion is initialized with all zeros.

        # Running costs:
        prog.AddQuadraticErrorCost(np.diag(q_cost), q0, q[:,n])
        prog.AddQuadraticErrorCost(np.diag(v_cost), [0]*nv, v[:,n])

    # Make a new autodiff context for this constraint (to maximize cache hits)
    ad_velocity_dynamics_context = [ad_plant.CreateDefaultContext() for i in range(N)]
    def velocity_dynamics_constraint(vars, context_index):
        h, q, v, qn = np.split(vars, [1, 1+nq, 1+nq+nv])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(q, ad_plant.GetPositions(ad_velocity_dynamics_context[context_index])):
                ad_plant.SetPositions(ad_velocity_dynamics_context[context_index], q)
            v_from_qdot = ad_plant.MapQDotToVelocity(ad_velocity_dynamics_context[context_index], (qn - q)/h)
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            v_from_qdot = plant.MapQDotToVelocity(context[context_index], (qn - q)/h)
        return v - v_from_qdot
    for n in range(N-1):
        prog.AddConstraint(
            partial(velocity_dynamics_constraint, context_index=n), 
            lb=[0]*nv, ub=[0]*nv, 
            vars=np.concatenate(([h[n]], q[:,n], v[:,n], q[:,n+1])))

    # Contact forces
    num_contacts = robot.get_num_contacts()
    contact_force = [prog.NewContinuousVariables(3, N-1, f"contact{contact}_contact_force") for contact in range(num_contacts)]
    for n in range(N-1):
        for contact in range(num_contacts):
            # Linear friction cone
            prog.AddLinearConstraint(contact_force[contact][0,n] <= mu*contact_force[contact][2,n])
            prog.AddLinearConstraint(-contact_force[contact][0,n] <= mu*contact_force[contact][2,n])
            prog.AddLinearConstraint(contact_force[contact][1,n] <= mu*contact_force[contact][2,n])
            prog.AddLinearConstraint(-contact_force[contact][1,n] <= mu*contact_force[contact][2,n])
            # normal force >=0, normal_force == 0 if not in_stance
            # max normal force assumed to be 4mg
            prog.AddBoundingBoxConstraint(0, in_stance[contact,n]*4*g*total_mass, contact_force[contact][2,n])            

    # Center of mass variables and constraints
    com = prog.NewContinuousVariables(3, N, "com")
    comdot = prog.NewContinuousVariables(3, N, "comdot")
    comddot = prog.NewContinuousVariables(3, N-1, "comddot")
    # Initial CoM x,y position == 0
    prog.AddBoundingBoxConstraint(0, 0, com[:2,0])
    # Initial CoM z vel == 0
    prog.AddBoundingBoxConstraint(0, 0, comdot[2,0])
    # CoM height
    prog.AddBoundingBoxConstraint(.125, np.inf, com[2,:])
    # CoM x velocity >= 0
    prog.AddBoundingBoxConstraint(0, np.inf, comdot[0,:])
    # CoM final x position
    if is_laterally_symmetric:
        prog.AddBoundingBoxConstraint(stride_length/2.0, stride_length/2.0, com[0,-1])
    else:
        prog.AddBoundingBoxConstraint(stride_length, stride_length, com[0,-1])
    # CoM dynamics
    for n in range(N-1):
        # Note: The original matlab implementation used backwards Euler (here and throughout),
        # which is a little more consistent with the LCP contact models.
        prog.AddConstraint(eq(com[:, n+1], com[:,n] + h[n]*comdot[:,n]))
        prog.AddConstraint(eq(comdot[:, n+1], comdot[:,n] + h[n]*comddot[:,n]))
        prog.AddConstraint(eq(total_mass*comddot[:,n], sum(contact_force[i][:,n] for i in range(4)) + total_mass*gravity))

    # Angular momentum (about the center of mass)
    H = prog.NewContinuousVariables(3, N, "H")
    Hdot = prog.NewContinuousVariables(3, N-1, "Hdot")
    prog.SetInitialGuess(H, np.zeros((3, N)))
    prog.SetInitialGuess(Hdot, np.zeros((3,N-1)))
    # Hdot = sum_i cross(p_FootiW-com, contact_force_i)
    def angular_momentum_constraint(vars, context_index):
        q, com, Hdot, contact_force = np.split(vars, [nq, nq+3, nq+6])
        contact_force = contact_force.reshape(3, 4, order='F')
        if isinstance(vars[0], AutoDiffXd):
            q = autoDiffToValueMatrix(q)
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(4):
                p_WF = plant.CalcPointsPositions(context[context_index], contact_frame[i], [0,0,0], plant.world_frame())
                Jq_WF = plant.CalcJacobianTranslationalVelocity(
                    context[context_index], JacobianWrtVariable.kQDot,
                    contact_frame[i], [0, 0, 0], plant.world_frame(), plant.world_frame())
                ad_p_WF = initializeAutoDiffGivenGradientMatrix(p_WF, np.hstack((Jq_WF, np.zeros((3, 18)))))
                torque = torque     + np.cross(ad_p_WF.reshape(3) - com, contact_force[:,i])
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(4):
                p_WF = plant.CalcPointsPositions(context[context_index], contact_frame[i], [0,0,0], plant.world_frame())
                torque += np.cross(p_WF.reshape(3) - com, contact_force[:,i])
        return Hdot - torque
    for n in range(N-1):
        prog.AddConstraint(eq(H[:,n+1], H[:,n] + h[n]*Hdot[:,n]))
        Fn = np.concatenate([contact_force[i][:,n] for i in range(4)])
        prog.AddConstraint(partial(angular_momentum_constraint, context_index=n), lb=np.zeros(3), ub=np.zeros(3), 
                           vars=np.concatenate((q[:,n], com[:,n], Hdot[:,n], Fn)))

    # com == CenterOfMass(q), H = SpatialMomentumInWorldAboutPoint(q, v, com)
    # Make a new autodiff context for this constraint (to maximize cache hits)
    com_constraint_context = [ad_plant.CreateDefaultContext() for i in range(N)]
    def com_constraint(vars, context_index):
        qv, com, H = np.split(vars, [nq+nv, nq+nv+3])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(qv, ad_plant.GetPositionsAndVelocities(com_constraint_context[context_index])):
                ad_plant.SetPositionsAndVelocities(com_constraint_context[context_index], qv)
            com_q = ad_plant.CalcCenterOfMassPositionInWorld(com_constraint_context[context_index])
            H_qv = ad_plant.CalcSpatialMomentumInWorldAboutPoint(com_constraint_context[context_index], com).rotational()
        else:
            if not np.array_equal(qv, plant.GetPositionsAndVelocities(context[context_index])):
                plant.SetPositionsAndVelocities(context[context_index], qv)
            com_q = plant.CalcCenterOfMassPositionInWorld(context[context_index])
            H_qv = plant.CalcSpatialMomentumInWorldAboutPoint(context[context_index], com).rotational()
        return np.concatenate((com_q - com, H_qv - H))
    for n in range(N):
        prog.AddConstraint(partial(com_constraint, context_index=n), 
            lb=np.zeros(6), ub=np.zeros(6), vars=np.concatenate((q[:,n], v[:,n], com[:,n], H[:,n])))

    # TODO: Add collision constraints

    # Kinematic constraints
    def fixed_position_constraint(vars, context_index, frame):
        q, qn = np.split(vars, [nq])
        if not np.array_equal(q, plant.GetPositions(context[context_index])):
            plant.SetPositions(context[context_index], q)
        if not np.array_equal(qn, plant.GetPositions(context[context_index+1])):
            plant.SetPositions(context[context_index+1], qn)
        p_WF = plant.CalcPointsPositions(context[context_index], frame, [0,0,0], plant.world_frame())
        p_WF_n = plant.CalcPointsPositions(context[context_index+1], frame, [0,0,0], plant.world_frame())
        if isinstance(vars[0], AutoDiffXd):
            J_WF = plant.CalcJacobianTranslationalVelocity(context[context_index], JacobianWrtVariable.kQDot,
                                                    frame, [0, 0, 0], plant.world_frame(), plant.world_frame())
            J_WF_n = plant.CalcJacobianTranslationalVelocity(context[context_index+1], JacobianWrtVariable.kQDot,
                                                    frame, [0, 0, 0], plant.world_frame(), plant.world_frame())
            return initializeAutoDiffGivenGradientMatrix(
                p_WF_n - p_WF, J_WF_n @ autoDiffToGradientMatrix(qn) - J_WF @ autoDiffToGradientMatrix(q))
        else:
            return p_WF_n - p_WF
    for i in range(4):
        for n in range(N):
            if in_stance[i, n]:
                # foot should be on the ground (world position z=0)
                prog.AddConstraint(PositionConstraint(
                    plant, plant.world_frame(), [-np.inf,-np.inf,0], [np.inf,np.inf,0], 
                    contact_frame[i], [0,0,0], context[n]), q[:,n])
                if n > 0 and in_stance[i, n-1]:
                    # feet should not move during stance.
                    prog.AddConstraint(partial(fixed_position_constraint, context_index=n-1, frame=contact_frame[i]),
                                       lb=np.zeros(3), ub=np.zeros(3), vars=np.concatenate((q[:,n-1], q[:,n])))
            else:
                min_clearance = 0.01
                prog.AddConstraint(PositionConstraint(plant, plant.world_frame(), [-np.inf,-np.inf,min_clearance], [np.inf,np.inf,np.inf],contact_frame[i],[0,0,0],context[n]), q[:,n])

    # Periodicity constraints
    if is_laterally_symmetric:
        # Joints
        def AddAntiSymmetricPair(a, b):
            prog.AddLinearEqualityConstraint(a[0] == -b[-1])
            prog.AddLinearEqualityConstraint(a[-1] == -b[0])
        def AddSymmetricPair(a, b):
            prog.AddLinearEqualityConstraint(a[0] == b[-1])
            prog.AddLinearEqualityConstraint(a[-1] == b[0])

        AddAntiSymmetricPair(q_view.front_left_hip_roll,        
                             q_view.front_right_hip_roll)
        AddSymmetricPair(q_view.front_left_hip_pitch,
                         q_view.front_right_hip_pitch)
        AddSymmetricPair(q_view.front_left_knee, q_view.front_right_knee)
        AddAntiSymmetricPair(q_view.back_left_hip_roll, 
                             q_view.back_right_hip_roll)
        AddSymmetricPair(q_view.back_left_hip_pitch, 
                         q_view.back_right_hip_pitch)
        AddSymmetricPair(q_view.back_left_knee, q_view.back_right_knee)               
        prog.AddLinearEqualityConstraint(q_view.body_y[0] == -q_view.body_y[-1])
        prog.AddLinearEqualityConstraint(q_view.body_z[0] == q_view.body_z[-1])
        # Body orientation must be in the xz plane:
        prog.AddBoundingBoxConstraint(0, 0, q_view.body_qx[[0,-1]])
        prog.AddBoundingBoxConstraint(0, 0, q_view.body_qz[[0,-1]])

        # Floating base velocity
        prog.AddLinearEqualityConstraint(v_view.body_vx[0] == v_view.body_vx[-1])
        prog.AddLinearEqualityConstraint(v_view.body_vy[0] == -v_view.body_vy[-1])
        prog.AddLinearEqualityConstraint(v_view.body_vz[0] == v_view.body_vz[-1])

        # CoM velocity
        prog.AddLinearEqualityConstraint(comdot[0,0] == comdot[0,-1])
        prog.AddLinearEqualityConstraint(comdot[1,0] == -comdot[1,-1])
        prog.AddLinearEqualityConstraint(comdot[2,0] == comdot[2,-1])

    else:
        # Everything except body_x is periodic
        q_selector = PositionView([True]*nq)
        q_selector.body_x = False
        prog.AddLinearConstraint(eq(q[q_selector,0], q[q_selector,-1]))
        prog.AddLinearConstraint(eq(v[:,0], v[:,-1]))

    # TODO: Set solver parameters (mostly to make the worst case solve times less bad)
    snopt = SnoptSolver().solver_id()
    prog.SetSolverOption(snopt, 'Iterations Limits', 1e5 if running_as_notebook else 1)
    prog.SetSolverOption(snopt, 'Major Iterations Limit', 200 if running_as_notebook else 1)
    prog.SetSolverOption(snopt, 'Major Feasibility Tolerance', 5e-6)
    prog.SetSolverOption(snopt, 'Major Optimality Tolerance', 1e-4)
    prog.SetSolverOption(snopt, 'Superbasics limit', 2000)
    prog.SetSolverOption(snopt, 'Linesearch tolerance', 0.9)
    #prog.SetSolverOption(snopt, 'Print file', 'snopt.out')

    # TODO a few more costs/constraints from 
    # from https://github.com/RobotLocomotion/LittleDog/blob/master/gaitOptimization.m 

    result = Solve(prog)
    print(result.get_solver_id().name())
    #print(result.is_success())  # We expect this to be false if iterations are limited.

    def HalfStrideToFullStride(a):
        b = PositionView(np.copy(a))

        b.body_y = -a.body_y
        # Mirror quaternion so that roll=-roll, yaw=-yaw
        b.body_qx = -a.body_qx
        b.body_qz = -a.body_qz

        b.front_left_hip_roll = -a.front_right_hip_roll
        b.front_right_hip_roll = -a.front_left_hip_roll
        b.back_left_hip_roll = -a.back_right_hip_roll
        b.back_right_hip_roll = -a.back_left_hip_roll

        b.front_left_hip_pitch = a.front_right_hip_pitch
        b.front_right_hip_pitch = a.front_left_hip_pitch
        b.back_left_hip_pitch = a.back_right_hip_pitch
        b.back_right_hip_pitch = a.back_left_hip_pitch

        b.front_left_knee = a.front_right_knee
        b.front_right_knee = a.front_left_knee
        b.back_left_knee = a.back_right_knee
        b.back_right_knee = a.back_left_knee

        return b


    # Animate trajectory
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    t_sol = np.cumsum(np.concatenate(([0],result.GetSolution(h))))
    q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))
    visualizer.start_recording()
    num_strides = 4
    t0 = t_sol[0]
    tf = t_sol[-1]
    T = tf*num_strides*(2.0 if is_laterally_symmetric else 1.0)
    for t in np.hstack((np.arange(t0, T, visualizer.draw_period), T)):
        context.SetTime(t)
        stride = (t - t0) // (tf - t0)
        ts = (t - t0) % (tf - t0)
        qt = PositionView(q_sol.value(ts))
        if is_laterally_symmetric:
            if stride % 2 == 1:
                qt = HalfStrideToFullStride(qt)
                qt.body_x += stride_length/2.0
            stride = stride // 2
        qt.body_x += stride*stride_length
        plant.SetPositions(plant_context, qt)
        diagram.Publish(context)

    visualizer.stop_recording()
    visualizer.publish_recording()

# Try them all!  The last two could use a little tuning.
littledog_walking_trot = partial(LittleDog, gait="walking_trot")
littledog_running_trot = partial(LittleDog, gait="running_trot")
littledog_rotary_gallop = partial(LittleDog, gait="rotary_gallop")
littledog_bound = partial(LittleDog, gait="bound")
# gait_optimization(littledog_walking_trot)

gait_optimization(Atlas)

import time
time.sleep(1e5)
