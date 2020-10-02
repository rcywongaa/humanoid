#!/usr/bin/python3

'''
This implements the paper
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics
by Hongkai Dai, Andrés Valenzuela and Russ Tedrake
'''

from load_atlas import load_atlas, set_atlas_initial_pose
from load_atlas import getSortedJointLimits, getActuatorIndex, getActuatorIndices, getJointValues
from load_atlas import JOINT_LIMITS, lfoot_full_contact_points, rfoot_full_contact_points, FLOATING_BASE_DOF, FLOATING_BASE_QUAT_DOF, NUM_ACTUATED_DOF, TOTAL_DOF, M
from pydrake.all import Quaternion
from pydrake.all import Multiplexer
from pydrake.all import PiecewisePolynomial, PiecewiseTrajectory, PiecewiseQuaternionSlerp, TrajectorySource
from pydrake.all import ConnectDrakeVisualizer, ConnectContactResultsToDrakeVisualizer, Simulator
from pydrake.all import DiagramBuilder, MultibodyPlant, AddMultibodyPlantSceneGraph, BasicVector, LeafSystem
from pydrake.all import MathematicalProgram, Solve, IpoptSolver, eq, le, ge, SolverOptions
from balance import HumanoidController
import numpy as np
import time
import pdb

mbp_time_step = 1.0e-3
N_d = 4 # friction cone approximated as a i-pyramid
N_f = 3 # contact force dimension

MAX_GROUND_PENETRATION = 1e-2

num_contact_points = lfoot_full_contact_points.shape[1]+rfoot_full_contact_points.shape[1]
mu = 1.0 # Coefficient of friction, same as in load_atlas.py
n = np.array([
    [0],
    [0],
    [1.0]])
d = np.array([
    [1.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -1.0],
    [0.0, 0.0, 0.0, 0.0]])
# Equivalent to v in balance.py
friction_cone_components = np.zeros((N_d, num_contact_points, N_f))
for i in range(N_d):
    for j in range(num_contact_points):
        friction_cone_components[i,j] = (n+mu*d)[:,i]

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

def toTauj(tau_k):
    return np.hstack([np.zeros((num_contact_points, 2)), np.reshape(tau_k, (num_contact_points, 1))])

def calcTrajectory(q_init, q_final, num_knot_points, max_time, pelvis_only=False):
    N = num_knot_points
    T = max_time
    plant_float = MultibodyPlant(mbp_time_step)
    load_atlas(plant_float)
    context_float = plant_float.CreateDefaultContext()
    plant_autodiff = plant_float.ToAutoDiffXd()
    context_autodiff = plant_autodiff.CreateDefaultContext()
    upright_context = plant_float.CreateDefaultContext()
    set_atlas_initial_pose(plant_float, upright_context)
    q_nom = plant_float.GetPositions(upright_context)

    def getPlantAndContext(q, v):
        assert(q.dtype == v.dtype)
        if q.dtype == np.object:
            plant_autodiff.SetPositions(context_autodiff, q)
            plant_autodiff.SetVelocities(context_autodiff, v)
            return plant_autodiff, context_autodiff
        else:
            plant_float.SetPositions(context_float, q)
            plant_float.SetVelocities(context_float, v)
            return plant_float, context_float


    # Returns contact positions in the shape [3, num_contact_points]
    def get_contact_positions(q, v):
        plant, context = getPlantAndContext(q, v)
        lfoot_full_contact_positions = plant.CalcPointsPositions(
                context, plant.GetFrameByName("l_foot"),
                lfoot_full_contact_points, plant.world_frame())
        rfoot_full_contact_positions = plant.CalcPointsPositions(
                context, plant.GetFrameByName("r_foot"),
                rfoot_full_contact_points, plant.world_frame())
        return np.concatenate([lfoot_full_contact_positions, rfoot_full_contact_positions], axis=1)

    sorted_joint_position_lower_limits = np.array([entry[1].lower for entry in getSortedJointLimits(plant_float)])
    sorted_joint_position_upper_limits = np.array([entry[1].upper for entry in getSortedJointLimits(plant_float)])
    sorted_joint_velocity_limits = np.array([entry[1].velocity for entry in getSortedJointLimits(plant_float)])

    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(rows=N, cols=plant_float.num_positions(), name="q")
    v = prog.NewContinuousVariables(rows=N, cols=plant_float.num_velocities(), name="v")
    dt = prog.NewContinuousVariables(N, name="dt")
    r = prog.NewContinuousVariables(rows=N, cols=3, name="r")
    rd = prog.NewContinuousVariables(rows=N, cols=3, name="rd")
    rdd = prog.NewContinuousVariables(rows=N, cols=3, name="rdd")
    contact_dim = 3*num_contact_points
    # The cols are ordered as
    # [contact1_x, contact1_y, contact1_z, contact2_x, contact2_y, contact2_z, ...]
    c = prog.NewContinuousVariables(rows=N, cols=contact_dim, name="c")
    F = prog.NewContinuousVariables(rows=N, cols=contact_dim, name="F")
    tau = prog.NewContinuousVariables(rows=N, cols=num_contact_points, name="tau") # We assume only z torque exists
    h = prog.NewContinuousVariables(rows=N, cols=3, name="h")
    hd = prog.NewContinuousVariables(rows=N, cols=3, name="hd")

    '''
    Slack for the complementary constraints
    Same value used in drake/multibody/optimization/static_equilibrium_problem.cc
    '''
    slack = 1e-3

    ''' Additional variables not explicitly stated '''
    # Friction cone scale
    beta = prog.NewContinuousVariables(rows=N, cols=num_contact_points*N_d, name="beta")

    g = np.array([0, 0, -9.81])
    for k in range(N):
        ''' Eq(7a) '''
        Fj = np.reshape(F[k], (num_contact_points, 3))
        (prog.AddLinearConstraint(eq(M*rdd[k], np.sum(Fj, axis=0) + M*g))
                .evaluator().set_description(f"Eq(7a)[{k}]"))
        ''' Eq(7b) '''
        cj = np.reshape(c[k], (num_contact_points, 3))
        tauj = toTauj(tau[k])
        (prog.AddConstraint(eq(hd[k], np.sum(np.cross(cj - r[k], Fj) + tauj, axis=0)))
                .evaluator().set_description(f"Eq(7b)[{k}]"))
        ''' Eq(7c) '''
        # https://stackoverflow.com/questions/63454077/how-to-obtain-centroidal-momentum-matrix/63456202#63456202
        def calc_h(q, v):
            plant, context = getPlantAndContext(q, v)
            return plant.CalcSpatialMomentumInWorldAboutPoint(context, plant.CalcCenterOfMassPosition(context)).rotational()
        def eq7c(q_v_h):
            q, v, h = np.split(q_v_h, [
                plant_float.num_positions(),
                plant_float.num_positions() + plant_float.num_velocities()])
            return calc_h(q, v) - h
        (prog.AddConstraint(eq7c, lb=[0]*3, ub=[0]*3, vars=np.concatenate([q[k], v[k], h[k]]))
            .evaluator().set_description(f"Eq(7c)[{k}]"))
        ''' Eq(7h) '''
        def calc_r(q, v):
            plant, context = getPlantAndContext(q, v)
            return plant.CalcCenterOfMassPosition(context)
        def eq7h(q_v_r):
            q, v, r = np.split(q_v_r, [
                plant_float.num_positions(),
                plant_float.num_positions() + plant_float.num_velocities()])
            return  calc_r(q, v) - r
        # COM position has dimension 3
        (prog.AddConstraint(eq7h, lb=[0]*3, ub=[0]*3, vars=np.concatenate([q[k], v[k], r[k]]))
                .evaluator().set_description(f"Eq(7h)[{k}]"))
        ''' Eq(7i) '''
        def eq7i(q_v_ck):
            q, v, ck = np.split(q_v_ck, [
                plant_float.num_positions(),
                plant_float.num_positions() + plant_float.num_velocities()])
            cj = np.reshape(ck, (num_contact_points, 3))
            # print(f"q = {q}\nv={v}\nck={ck}")
            contact_positions = get_contact_positions(q, v).T
            return (contact_positions - cj).flatten()
        # np.concatenate cannot work q, cj since they have different dimensions
        (prog.AddConstraint(eq7i, lb=np.zeros(c[k].shape).flatten(), ub=np.zeros(c[k].shape).flatten(), vars=np.concatenate([q[k], v[k], c[k]]))
                .evaluator().set_description(f"Eq(7i)[{k}]"))
        ''' Eq(7j) '''
        (prog.AddBoundingBoxConstraint([-10, -10, -MAX_GROUND_PENETRATION]*num_contact_points, [10, 10, 10]*num_contact_points, c[k])
                .evaluator().set_description(f"Eq(7j)[{k}]"))
        ''' Eq(7k) '''
        ''' Constrain admissible posture '''
        (prog.AddBoundingBoxConstraint(sorted_joint_position_lower_limits, sorted_joint_position_upper_limits,
                q[k, FLOATING_BASE_QUAT_DOF:]).evaluator().set_description(f"Eq(7k)[{k}] joint position"))
        ''' Constrain velocities '''
        (prog.AddBoundingBoxConstraint(-sorted_joint_velocity_limits, sorted_joint_velocity_limits,
            v[k, FLOATING_BASE_DOF:]).evaluator().set_description(f"Eq(7k)[{k}] joint velocity"))
        ''' Constrain forces within friction cone '''
        beta_k = np.reshape(beta[k], (num_contact_points, N_d))
        for i in range(num_contact_points):
            beta_v = beta_k[i].dot(friction_cone_components[:,i,:])
            (prog.AddLinearConstraint(eq(Fj[i], beta_v))
                    .evaluator().set_description(f"Eq(7k)[{k}] friction cone constraint"))
        ''' Constrain beta positive '''
        for b in beta_k.flat:
            (prog.AddLinearConstraint(b >= 0.0)
                    .evaluator().set_description(f"Eq(7k)[{k}] beta >= 0 constraint"))
        ''' Constrain torques - assume torque linear to friction cone'''
        friction_torque_coefficient = 0.1
        for i in range(num_contact_points):
            max_torque = friction_torque_coefficient * np.sum(beta_k[i])
            (prog.AddLinearConstraint(le(tau[k][i], np.array([max_torque])))
                    .evaluator().set_description(f"Eq(7k)[{k}] friction torque upper limit"))
            (prog.AddLinearConstraint(ge(tau[k][i], np.array([-max_torque])))
                    .evaluator().set_description(f"Eq(7k)[{k}] friction torque lower limit"))

        ''' Assume flat ground for now... '''
        def get_contact_positions_z(q, v):
            return get_contact_positions(q, v)[2,:]
        ''' Eq(8a) '''
        def eq8a_lhs(q_v_F):
            q, v, F = np.split(q_v_F, [
                plant_float.num_positions(),
                plant_float.num_positions() + plant_float.num_velocities()])
            Fj = np.reshape(F, (num_contact_points, 3))
            return [Fj[:,2].dot(get_contact_positions_z(q, v))] # Constraint functions must output vectors
        (prog.AddConstraint(eq8a_lhs, lb=[-slack], ub=[slack], vars=np.concatenate([q[k], v[k], F[k]]))
                .evaluator().set_description(f"Eq(8a)[{k}]"))
        ''' Eq(8b) '''
        def eq8b_lhs(q_v_tau):
            q, v, tau = np.split(q_v_tau, [
                plant_float.num_positions(),
                plant_float.num_positions() + plant_float.num_velocities()])
            tauj = toTauj(tau)
            return (tauj**2).T.dot(get_contact_positions_z(q, v)) # Outputs per axis sum of torques of all contact points
        (prog.AddConstraint(eq8b_lhs, lb=[-slack]*3, ub=[slack]*3, vars=np.concatenate([q[k], v[k], tau[k]]))
                .evaluator().set_description(f"Eq(8b)[{k}]"))
        ''' Eq(8c) '''
        (prog.AddLinearConstraint(ge(Fj[:,2], 0.0))
                .evaluator().set_description(f"Eq(8c)[{k}] contact force greater than zero"))
        def eq8c_2(q_v):
            q, v = np.split(q_v, [plant_float.num_positions()])
            return get_contact_positions_z(q, v)
        (prog.AddConstraint(eq8c_2, lb=[-MAX_GROUND_PENETRATION]*num_contact_points, ub=[float('inf')]*num_contact_points, vars=np.concatenate([q[k], v[k]]))
                .evaluator().set_description(f"Eq(8c)[{k}] z position greater than zero"))

    for k in range(1, N):
        ''' Eq(7d) '''
        def eq7d(q_qprev_v_dt):
            q, qprev, v, dt = np.split(q_qprev_v_dt, [
                plant_float.num_positions(),
                plant_float.num_positions() + plant_float.num_positions(),
                plant_float.num_positions() + plant_float.num_positions() + plant_float.num_velocities()])
            plant, context = getPlantAndContext(q, v)
            qd = plant.MapVelocityToQDot(context, v*dt[0])
            return q - qprev - qd
        # dt[k] must be converted to an array
        (prog.AddConstraint(eq7d, lb=[0.0]*plant_float.num_positions(), ub=[0.0]*plant_float.num_positions(),
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

        Fj = np.reshape(F[k], (num_contact_points, 3))
        cj = np.reshape(c[k], (num_contact_points, 3))
        cj_prev = np.reshape(c[k-1], (num_contact_points, 3))
        for i in range(num_contact_points):
            ''' Assume flat ground for now... '''
            ''' Eq(9a) '''
            def eq9a_lhs(F_c_cprev):
                F, c, c_prev = np.split(F_c_cprev, [
                    contact_dim,
                    contact_dim + contact_dim])
                Fj = np.reshape(F, (num_contact_points, 3))
                cj = np.reshape(c, (num_contact_points, 3))
                cj_prev = np.reshape(c_prev, (num_contact_points, 3))
                return [Fj[i,2] * (cj[i] - cj_prev[i]).dot(np.array([1.0, 0.0, 0.0]))]
            (prog.AddConstraint(eq9a_lhs, ub=[slack], lb=[-slack], vars=np.concatenate([F[k], c[k], c[k-1]]))
                    .evaluator().set_description("Eq(9a)[{k}][{i}]"))
            ''' Eq(9b) '''
            def eq9b_lhs(F_c_cprev):
                F, c, c_prev = np.split(F_c_cprev, [
                    contact_dim,
                    contact_dim + contact_dim])
                Fj = np.reshape(F, (num_contact_points, 3))
                cj = np.reshape(c, (num_contact_points, 3))
                cj_prev = np.reshape(c_prev, (num_contact_points, 3))
                return [Fj[i,2] * (cj[i] - cj_prev[i]).dot(np.array([0.0, 1.0, 0.0]))]
            (prog.AddConstraint(eq9b_lhs, ub=[slack], lb=[-slack], vars=np.concatenate([F[k], c[k], c[k-1]]))
                    .evaluator().set_description("Eq(9b)[{k}][{i}]"))
    # ''' Eq(10) '''
    # Q_q = 0.1 * np.identity(plant_float.num_velocities())
    # Q_v = 1.0 * np.identity(plant_float.num_velocities())
    # for k in range(N):
        # def pose_error_cost(q_v_dt):
            # q, v, dt = np.split(q_v_dt, [
                # plant_float.num_positions(),
                # plant_float.num_positions() + plant_float.num_velocities()])
            # plant, context = getPlantAndContext(q, v)
            # q_err = plant.MapQDotToVelocity(context, q-q_nom)
            # return (dt*(q_err.dot(Q_q).dot(q_err)))[0] # AddCost requires cost function to return scalar, not array
        # prog.AddCost(pose_error_cost, vars=np.concatenate([q[k], v[k], [dt[k]]])) # np.concatenate requires items to have compatible shape
        # prog.AddCost(dt[k]*(
                # + v[k].dot(Q_v).dot(v[k])
                # + rdd[k].dot(rdd[k])))

    ''' Additional constraints not explicitly stated '''
    ''' Constrain initial pose '''
    (prog.AddLinearConstraint(eq(q[0], q_init))
            .evaluator().set_description("initial pose"))
    ''' Constrain initial velocity '''
    prog.AddLinearConstraint(eq(v[0], 0.0))
    ''' Constrain final pose '''
    if pelvis_only:
        (prog.AddLinearConstraint(eq(q[-1, 4:7], q_final[4:7]))
                .evaluator().set_description("final pose"))
    else:
        (prog.AddLinearConstraint(eq(q[-1], q_final))
                .evaluator().set_description("final pose"))
    ''' Constrain final velocity '''
    prog.AddLinearConstraint(eq(v[-1], 0.0))
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
        get_contact_positions(q_guess[i], v_guess[i]).T.flatten() for i in range(N)])
    for i in range(N):
        # pdb.set_trace()
        assert((eq7i(np.concatenate([q_guess[i], v_guess[i], c_guess[i]])) == 0.0).all())
    prog.SetDecisionVariableValueInVector(c, c_guess, initial_guess)

    r_guess = np.array([
        calc_r(q_guess[i], v_guess[i]) for i in range(N)])
    prog.SetDecisionVariableValueInVector(r, r_guess, initial_guess)

    h_guess = np.array([
        calc_h(q_guess[i], v_guess[i]) for i in range(N)])
    prog.SetDecisionVariableValueInVector(h, h_guess, initial_guess)

    solver = IpoptSolver()
    options = SolverOptions()
    # options.SetOption(solver.solver_id(), "max_iter", 50000)
    # This doesn't seem to do anything...
    # options.SetOption(CommonSolverOption.kPrintToConsole, True)
    start_solve_time = time.time()
    print(f"Start solving...")
    result = solver.Solve(prog, initial_guess, options) # Currently takes around 30 mins
    print(f"Solve time: {time.time() - start_solve_time}s  Cost: {result.get_optimal_cost()}")
    if not result.is_success():
        print(f"FAILED")
        print(result.GetInfeasibleConstraintNames(prog))
        q_sol = result.GetSolution(q)
        v_sol = result.GetSolution(v)
        dt_sol = result.GetSolution(dt)
        r_sol = result.GetSolution(r)
        rd_sol = result.GetSolution(rd)
        rdd_sol = result.GetSolution(rdd)
        c_sol = result.GetSolution(c)
        F_sol = result.GetSolution(F)
        tau_sol = result.GetSolution(tau)
        h_sol = result.GetSolution(h)
        hd_sol = result.GetSolution(hd)
        beta_sol = result.GetSolution(beta)
        pdb.set_trace()
    print(f"SUCCESS")
    r_sol = result.GetSolution(r)
    rd_sol = result.GetSolution(rd)
    rdd_sol = result.GetSolution(rdd)
    q_sol = result.GetSolution(q)
    v_sol = result.GetSolution(v)
    dt_sol = result.GetSolution(dt)

    return r_sol, rd_sol, rdd_sol, q_sol, v_sol, dt_sol

def main():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(mbp_time_step))
    load_atlas(plant, add_ground=True)
    plant_context = plant.CreateDefaultContext()
    set_atlas_initial_pose(plant, plant_context)

    q_init = plant.GetPositions(plant_context)
    q_init[6] = 1.0 # Avoid initializing with ground penetration
    q_final = q_init.copy()
    # q_final[4] = 0.1 # x position of pelvis
    q_final[6] = 0.90 # z position of pelvis (to make sure final pose touches ground)

    num_knot_points = 6
    # max_time = 0.14278
    max_time = 0.15

    print(f"Starting pos: {q_init}\nFinal pos: {q_final}")
    r_traj, rd_traj, rdd_traj, q_traj, v_traj, dt_traj = (
            calcTrajectory(q_init, q_final, num_knot_points, max_time, pelvis_only=True))

    controller = builder.AddSystem(HumanoidController(is_wbc=True))
    controller.set_name("HumanoidController")

    ''' Connect atlas plant to controller '''
    builder.Connect(plant.get_state_output_port(), controller.GetInputPort("q_v"))
    builder.Connect(controller.GetOutputPort("tau"), plant.get_actuation_input_port())

    r_traj = np.array([[ 1.33884302e-03,  1.04911005e-03,  1.29087388e+00],
        [ 6.94418259e-03, -8.23044969e-04,  1.19479193e+00],
        [ 7.88826379e-03, -1.26066513e-03,  1.17813942e+00],
        [ 8.20776635e-03, -1.33447847e-03,  1.17348443e+00],
        [ 8.36268118e-03, -1.35747624e-03,  1.17133799e+00],
        [ 8.36268118e-03, -1.35747624e-03,  1.17133799e+00]])
    rd_traj = np.array([[ 2.04789298e-01, -4.76702934e-02, -3.34777500e+00],
       [ 8.10624419e-02, -4.78027396e-02, -1.55205167e+00],
       [ 5.52423379e-02, -1.53800883e-02, -8.52208449e-01],
       [ 3.96022928e-02, -6.53146809e-03, -5.29629198e-01],
       [ 1.21963912e-02, -1.15826851e-03, -1.88074306e-01],
       [-1.21963912e-02,  1.15826851e-03,  1.88074306e-01]])
    rdd_traj = np.array([[-1.15194592e-11, -1.16752884e-11, -9.80990798e+00],
       [-3.15480772e+00, -3.37713584e-03,  4.57876485e+01],
       [-1.86393054e+00,  2.34056262e+00,  5.05210655e+01],
       [-2.32138096e+00,  1.31336056e+00,  4.78789752e+01],
       [-4.58183899e+00,  8.98315101e-01,  5.71026467e+01],
       [-3.35942169e+00,  3.19038008e-01,  5.18039223e+01]])
    q_traj = np.array([[ 1.00000000e+00, -1.78670980e-31, -2.62178581e-31,
         5.91324996e-33,  0.00000000e+00,  3.33036112e-31,
         1.00000000e+00, -9.85832891e-34, -6.46995948e-33,
         3.89942259e-42,  0.00000000e+00,  7.19865224e-39,
        -1.73327230e-32,  1.29075436e-32,  8.72063406e-33,
        -2.31156703e-33, -1.45359427e-33,  1.04215903e-40,
        -1.78219316e-33,  4.56437991e-33, -4.53142199e-33,
         8.62865659e-35, -4.82039009e-35, -2.00020649e-34,
         0.00000000e+00,  1.31988898e-40, -1.55073620e-35,
        -1.54573393e-33, -3.64230598e-34, -3.03977877e-34,
         9.24789416e-35,  6.60052628e-41,  1.03029660e-36,
         1.80549733e-35,  1.57817636e-35, -7.85423798e-37,
        -3.14169232e-36],
       [ 9.54565540e-01, -5.89599891e-02, -9.86119164e-02,
         1.73693925e-01,  2.31315477e-03, -2.37029564e-03,
         9.15743897e-01, -2.65528081e-01,  2.16624540e-01,
         3.62444117e-02,  3.08488892e-01,  1.30788842e-01,
         1.02811279e-01,  1.50301354e-01,  3.75267559e-01,
         2.48617500e-02, -2.02808801e-01,  3.70092031e-01,
        -1.10342256e-01, -4.02581701e-01,  2.28761557e-01,
        -1.65879729e-01,  2.12611036e-01,  1.79404371e-01,
        -1.08951862e-01,  3.77165385e-01, -1.59765142e-02,
         7.00140991e-02,  7.57043385e-02,  1.14029729e-02,
         2.81680149e-03,  3.71862175e-01, -5.08263792e-04,
        -1.95842167e-01, -4.13525151e-04, -8.57421317e-02,
        -2.24303567e-01],
       [ 9.39828458e-01, -8.04798407e-02, -1.30695769e-01,
         2.11215713e-01,  4.43054107e-03, -2.16469519e-03,
         9.04737769e-01, -3.28819040e-01,  2.34805319e-01,
         2.91760209e-02,  4.03066537e-01,  1.99183685e-01,
         1.54763167e-01,  1.96834560e-01,  4.74319748e-01,
         4.66895475e-02, -2.62141089e-01,  4.41590142e-01,
        -7.81979124e-02, -5.14046285e-01,  2.95749062e-01,
        -2.33739380e-01,  2.70048044e-01,  2.06345221e-01,
        -1.43273008e-01,  4.53087471e-01, -1.68421963e-02,
         7.65368881e-02,  8.54935816e-02,  4.31374963e-03,
         1.29767983e-02,  4.44638453e-01, -5.83107307e-04,
        -2.17564590e-01,  4.98680393e-04, -9.40640975e-02,
        -2.49131961e-01],
       [ 9.35804446e-01, -8.32538430e-02, -1.35086419e-01,
         2.24328722e-01,  6.69884376e-03, -3.70598705e-04,
         9.00588724e-01, -3.54149444e-01,  2.46087659e-01,
         3.14812926e-02,  4.10375602e-01,  2.08691879e-01,
         1.61653772e-01,  2.02532045e-01,  5.01442984e-01,
         5.54528972e-02, -2.80298822e-01,  4.56564357e-01,
        -7.28397604e-02, -5.55195308e-01,  3.14885012e-01,
        -2.46363905e-01,  2.78548097e-01,  2.16800645e-01,
        -1.47063187e-01,  4.69314552e-01, -1.69769313e-02,
         7.77012973e-02,  8.84687984e-02,  4.29665452e-03,
         1.43349393e-02,  4.60044958e-01, -6.14074354e-04,
        -2.21504437e-01,  5.86833515e-04, -9.56352493e-02,
        -2.53740918e-01],
       [ 9.30787472e-01, -8.60090766e-02, -1.39603607e-01,
         2.40160413e-01, -1.34950513e-32, -2.25349893e-31,
         9.00000000e-01, -3.80405873e-01,  2.51415394e-01,
         3.24581762e-02,  4.47289634e-01,  2.12445552e-01,
         1.56643748e-01,  2.07412145e-01,  4.99212273e-01,
         2.87433345e-02, -2.93885947e-01,  4.67243994e-01,
        -8.17730882e-02, -5.61126402e-01,  3.01615157e-01,
        -2.52093855e-01,  2.80369579e-01,  2.23356905e-01,
        -1.30302778e-01,  4.80492396e-01, -1.72706783e-02,
         8.07368472e-02,  9.33137388e-02,  4.93758571e-03,
         1.41747775e-02,  4.70710380e-01, -6.13887699e-04,
        -2.24261451e-01,  5.27689345e-04, -9.67562795e-02,
        -2.56888553e-01],
       [ 9.30787472e-01, -8.60090766e-02, -1.39603607e-01,
         2.40160413e-01,  0.00000000e+00,  0.00000000e+00,
         9.00000000e-01, -3.80405873e-01,  2.51415394e-01,
         3.24581762e-02,  4.47289634e-01,  2.12445552e-01,
         1.56643748e-01,  2.07412145e-01,  4.99212273e-01,
         2.87433345e-02, -2.93885947e-01,  4.67243994e-01,
        -8.17730882e-02, -5.61126402e-01,  3.01615157e-01,
        -2.52093855e-01,  2.80369579e-01,  2.23356905e-01,
        -1.30302778e-01,  4.80492396e-01, -1.72706783e-02,
         8.07368472e-02,  9.33137388e-02,  4.93758571e-03,
         1.41747775e-02,  4.70710380e-01, -6.13887699e-04,
        -2.24261451e-01,  5.27689345e-04, -9.67562795e-02,
        -2.56888553e-01]])
    v_traj = np.array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00, -1.51768810e-34,  4.11030196e-34,
         2.57979941e-33, -1.25627322e-33,  1.70137131e-33,
         9.18354962e-41, -1.34309413e-39,  2.20261048e-34,
        -2.91700744e-33, -3.03128884e-41,  0.00000000e+00,
        -6.55079124e-34,  1.49994357e-34,  9.61112329e-34,
         0.00000000e+00, -1.01979682e-34, -1.88740828e-35,
        -1.79007471e-40, -3.98722112e-35, -6.75912696e-36,
        -5.18175832e-35,  3.80011126e-34,  0.00000000e+00,
        -1.72511199e-42, -4.15727238e-36,  1.34462161e-36,
         0.00000000e+00, -1.63534602e-37,  1.38990109e-37],
       [-3.14985495e+00, -5.26820371e+00,  9.27935499e+00,
         5.89812005e-02, -6.04381878e-02, -2.14837597e+00,
        -6.77047866e+00,  5.52352813e+00,  9.24165969e-01,
         7.86590048e+00,  3.33487540e+00,  2.62149889e+00,
         3.83240864e+00,  9.56863391e+00,  6.33928989e-01,
        -5.17125216e+00,  9.43666744e+00, -2.81352498e+00,
        -1.02650944e+01,  5.83299980e+00, -4.22962860e+00,
         5.42119116e+00,  4.57448215e+00, -2.77807248e+00,
         9.61702499e+00, -4.07371785e-01,  1.78523101e+00,
         1.93032167e+00,  2.90754879e-01,  7.18232675e-02,
         9.48180287e+00, -1.29597937e-02, -4.99361577e+00,
        -1.05441323e-02, -2.18626697e+00, -5.71932923e+00],
       [-2.96485983e+00, -5.09941386e+00,  5.78854041e+00,
         1.52852251e-01,  1.48421148e-02, -7.94522656e-01,
        -4.56891855e+00,  1.31245445e+00, -5.10260900e-01,
         6.82747686e+00,  4.93736342e+00,  3.75036105e+00,
         3.35919108e+00,  7.15049023e+00,  1.57572946e+00,
        -4.28314554e+00,  5.16138560e+00,  2.32047183e+00,
        -8.04653004e+00,  4.83576891e+00, -4.89872842e+00,
         4.14632693e+00,  1.94483622e+00, -2.47761330e+00,
         5.48074848e+00, -6.24928312e-02,  4.70874389e-01,
         7.06676827e-01, -5.11764782e-01,  7.33441211e-01,
         5.25365538e+00, -5.40288737e-03, -1.56812256e+00,
         6.58513141e-02, -6.00755379e-01, -1.79233985e+00],
       [-1.16008112e+00, -1.30537468e+00,  4.10831041e+00,
         3.36673880e-01,  2.66289604e-01, -6.15823934e-01,
        -3.75967698e+00,  1.67458659e+00,  3.42161019e-01,
         1.08485128e+00,  1.41125814e+00,  1.02274114e+00,
         8.45651996e-01,  4.02577889e+00,  1.30070426e+00,
        -2.69506992e+00,  2.22255490e+00,  7.95286193e-01,
        -6.10756286e+00,  2.84026230e+00, -1.87380099e+00,
         1.26162436e+00,  1.55185117e+00, -5.62559056e-01,
         2.40851204e+00, -1.99981008e-02,  1.72827967e-01,
         4.41597945e-01, -2.53734982e-03,  2.01582717e-01,
         2.28671770e+00, -4.59629837e-03, -5.84773722e-01,
         1.30841681e-02, -2.33198932e-01, -6.84086688e-01],
       [-1.44916382e+00, -1.47856112e+00,  5.60652615e+00,
        -1.11994212e+00,  6.19583189e-02, -9.84255167e-02,
        -4.38966506e+00,  8.90714088e-01,  1.63319684e-01,
         6.17145002e+00,  6.27555512e-01, -8.37597702e-01,
         8.15876405e-01, -3.72939977e-01, -4.46542199e+00,
        -2.27155531e+00,  1.78546852e+00, -1.49351296e+00,
        -9.91586322e-01, -2.21851260e+00, -9.57958120e-01,
         3.04523327e-01,  1.09610427e+00,  2.80207877e+00,
         1.86876096e+00, -4.91099169e-02,  5.07496564e-01,
         8.09998405e-01,  1.07153692e-01, -2.67765468e-02,
         1.78309205e+00,  3.12058553e-05, -4.60929681e-01,
        -9.88798215e-03, -1.87418747e-01, -5.26235425e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -1.06263720e-32,  0.00000000e+00, -4.62779636e-34,
        -5.16679873e-33,  1.93223463e-33,  5.04292954e-34,
        -2.12247278e-32,  2.80994520e-35,  6.07763349e-33,
         9.41199104e-34,  1.63803733e-34,  7.71155108e-34,
         2.44767128e-34,  1.46314913e-34,  7.54955334e-35,
        -2.10050050e-35,  3.98722952e-35, -6.62077770e-34,
         2.61368287e-34,  5.83460875e-40,  1.89642758e-34,
         1.79206180e-35, -2.93656052e-36, -6.61839833e-44,
         1.02680202e-35, -1.96355950e-37, -1.96355950e-37]])
    dt_traj = np.array([7.49833686e-41, 3.92185094e-02, 1.38525033e-02, 6.73738838e-03,
       5.98141963e-03, 7.26100640e-03])

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
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(0.1)
    simulator.AdvanceTo(5.0)

if __name__ == "__main__":
    main()
