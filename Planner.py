'''
Adapted from http://underactuated.mit.edu/humanoids.html#example1

Implements paper:
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics
by Hongkai Dai, Andrés Valenzuela and Russ Tedrake
'''
from LittleDog import LittleDog
from Atlas import Atlas

import pdb
import time
from functools import partial
import numpy as np
from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, Parser, ConnectMeshcatVisualizer, RigidTransform, Simulator, PidController
from pydrake.all import (
    MultibodyPlant, JointIndex, RotationMatrix, PiecewisePolynomial, JacobianWrtVariable,
    MathematicalProgram, Solve, eq, AutoDiffXd, ExtractGradient, SnoptSolver,
    InitializeAutoDiff, ExtractValue, ExtractGradient,
    AddUnitQuaternionConstraintOnPlant, PositionConstraint, OrientationConstraint
)

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess()

# Need this because a==b returns True even if a = AutoDiffXd(1, [1, 2]), b= AutoDiffXd(2, [3, 4])
# That's the behavior of AutoDiffXd in C++, also.
def autoDiffArrayEqual(a,b):
    return np.array_equal(a, b) and np.array_equal(ExtractGradient(a), ExtractGradient(b))

def gait_optimization(robot_ctor):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    robot = robot_ctor(plant)
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

    PositionView = robot.PositionView()
    VelocityView = robot.VelocityView()

    mu = 1 # rubber on rubber
    total_mass = robot.get_total_mass(context)
    gravity = plant.gravity_field().gravity_vector()
    g = 9.81
    # max normal force assumed to be 4mg
    max_contact_force = 4*g*total_mass

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
    q_cost = robot.get_position_cost()
    v_cost = robot.get_velocity_cost()
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
                                                 robot.max_body_rotation(), context[n]), q[:,n])
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
    '''
    Ordered as follows
    [[[ contact0_x_t0, ... , contact0_x_tn],
      [ contact0_y_t0, ... , contact0_y_tn],
      [ contact0_z_t0, ... , contact0_z_tn]],
     ...
     [[ contactn_x_t0, ... , contactn_x_tn],
      [ contactn_y_t0, ... , contactn_y_tn],
      [ contactn_z_t0, ... , contactn_z_tn]]]
    '''
    normalized_contact_force = [prog.NewContinuousVariables(3, N-1, f"contact{contact}_normalized_contact_force") for contact in range(num_contacts)]
    for n in range(N-1):
        for contact in range(num_contacts):
            # Linear friction cone
            prog.AddLinearConstraint(normalized_contact_force[contact][0,n] <= mu*normalized_contact_force[contact][2,n])
            prog.AddLinearConstraint(-normalized_contact_force[contact][0,n] <= mu*normalized_contact_force[contact][2,n])
            prog.AddLinearConstraint(normalized_contact_force[contact][1,n] <= mu*normalized_contact_force[contact][2,n])
            prog.AddLinearConstraint(-normalized_contact_force[contact][1,n] <= mu*normalized_contact_force[contact][2,n])
            # normal force >=0, normal_force == 0 if not in_stance
            prog.AddBoundingBoxConstraint(0.0, in_stance[contact,n], normalized_contact_force[contact][2,n])

            prog.SetInitialGuess(normalized_contact_force[contact][2,n], 0.1*in_stance[contact,n])

    # Center of mass variables and constraints
    com = prog.NewContinuousVariables(3, N, "com")
    comdot = prog.NewContinuousVariables(3, N, "comdot")
    comddot = prog.NewContinuousVariables(3, N-1, "comddot")
    # Initial CoM x,y position == 0
    prog.AddBoundingBoxConstraint(0, 0, com[:2,0])
    # Initial CoM z vel == 0
    prog.AddBoundingBoxConstraint(0, 0, comdot[2,0])
    # CoM height
    prog.AddBoundingBoxConstraint(robot.min_com_height(), np.inf, com[2,:])
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
        prog.AddConstraint(eq(total_mass*comddot[:,n],
            sum(max_contact_force*normalized_contact_force[i][:,n] for i in range(num_contacts)) + total_mass*gravity))

    # Angular momentum (about the center of mass)
    H = prog.NewContinuousVariables(3, N, "H")
    Hdot = prog.NewContinuousVariables(3, N-1, "Hdot")
    prog.SetInitialGuess(H, np.zeros((3, N)))
    prog.SetInitialGuess(Hdot, np.zeros((3,N-1)))
    # Hdot = sum_i cross(p_FootiW-com, contact_force_i)
    def angular_momentum_constraint(vars, context_index, active_contacts):
        q, com, Hdot, normalized_contact_force = np.split(vars, [nq, nq+3, nq+6])
        '''
        [[ contact0_x_tn, ... , contactn_x_tn ],
         [ contact0_y_tn, ... , contactn_y_tn ],
         [ contact0_z_tn, ... , contactn_z_tn ]]
        '''
        contact_force = max_contact_force*(normalized_contact_force.reshape(3, num_contacts, order='F'))
        if isinstance(vars[0], AutoDiffXd):
            q = ExtractValue(q)
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3, dtype='object')
            for contact in active_contacts:
                p_WF = plant.CalcPointsPositions(context[context_index], contact_frame[contact], [0,0,0], plant.world_frame())
                Jq_WF = plant.CalcJacobianTranslationalVelocity(
                    context[context_index], JacobianWrtVariable.kQDot,
                    contact_frame[contact], [0, 0, 0], plant.world_frame(), plant.world_frame())
                ad_p_WF = InitializeAutoDiff(p_WF, np.hstack((Jq_WF, np.zeros((3, 18)))))
                torque += np.cross(ad_p_WF.reshape(3) - com, contact_force[:,contact])
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for contact in active_contacts:
                p_WF = plant.CalcPointsPositions(context[context_index], contact_frame[contact], [0,0,0], plant.world_frame())
                torque += np.cross(p_WF.reshape(3) - com, contact_force[:,contact])
        return Hdot - torque
    for n in range(N-1):
        prog.AddConstraint(eq(H[:,n+1], H[:,n] + h[n]*Hdot[:,n]))
        active_contacts = np.where(in_stance[:,n])[0]
        '''
        [ contact0_x_tn, contact0_y_tn, contact0_z_tn, ... , contactn_x_tn, contactn_y_tn, contactn_z_tn ]
        '''
        Fn = np.concatenate([normalized_contact_force[i][:,n] for i in range(num_contacts)])
        prog.AddConstraint(partial(angular_momentum_constraint, context_index=n, active_contacts=active_contacts),
                lb=np.zeros(3), ub=np.zeros(3),
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
            return InitializeAutoDiff(
                p_WF_n - p_WF, J_WF_n @ ExtractGradient(qn) - J_WF @ ExtractGradient(q))
        else:
            return p_WF_n - p_WF
    for i in range(robot.get_num_contacts()):
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
        robot.add_periodic_constraints(prog, q_view, v_view)

        # CoM velocity
        prog.AddLinearEqualityConstraint(comdot[0,0] == comdot[0,-1])
        prog.AddLinearEqualityConstraint(comdot[1,0] == -comdot[1,-1])
        prog.AddLinearEqualityConstraint(comdot[2,0] == comdot[2,-1])
    else:
        # Everything except body_x is periodic
        q_selector = robot.get_periodic_view()
        prog.AddLinearConstraint(eq(q[q_selector,0], q[q_selector,-1]))
        prog.AddLinearConstraint(eq(v[:,0], v[:,-1]))

    # TODO: Set solver parameters (mostly to make the worst case solve times less bad)
    snopt = SnoptSolver().solver_id()
    prog.SetSolverOption(snopt, 'Iterations Limits', 2e6)
    prog.SetSolverOption(snopt, 'Major Iterations Limit', 500)
    prog.SetSolverOption(snopt, 'Major Feasibility Tolerance', 5e-6)
    prog.SetSolverOption(snopt, 'Major Optimality Tolerance', 1e-4)
    prog.SetSolverOption(snopt, 'Superbasics limit', 4000)
    prog.SetSolverOption(snopt, 'Linesearch tolerance', 0.9)
    # prog.SetSolverOption(snopt, 'Scale option', 2)
    prog.SetSolverOption(snopt, 'Print file', 'snopt.out')

    # TODO a few more costs/constraints from
    # from https://github.com/RobotLocomotion/LittleDog/blob/master/gaitOptimization.m

    now = time.time()
    result = Solve(prog)
    print(f"{time.time() - now}s - {result.get_solver_id().name()}: {result.is_success()}, Cost: {result.get_optimal_cost()}")
    #print(result.is_success())  # We expect this to be false if iterations are limited.

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
                qt = robot.HalfStrideToFullStride(qt)
                robot.increment_periodic_view(qt, stride_length/2.0)
            stride = stride // 2
        robot.increment_periodic_view(qt, stride*stride_length)
        plant.SetPositions(plant_context, np.array(qt))
        diagram.Publish(context)

    visualizer.stop_recording()
    visualizer.publish_recording()

# Try them all!  The last two could use a little tuning.
littledog_walking_trot = partial(LittleDog, gait="walking_trot")
littledog_running_trot = partial(LittleDog, gait="running_trot")
littledog_rotary_gallop = partial(LittleDog, gait="rotary_gallop")
littledog_bound = partial(LittleDog, gait="bound")

# gait_optimization(littledog_walking_trot)
# gait_optimization(littledog_rotary_gallop)

gait_optimization(partial(Atlas, simplified=True))

time.sleep(1e5)
