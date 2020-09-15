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
from pydrake.all import PiecewisePolynomial, PiecewiseTrajectory, PiecewiseQuaternionSlerp
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

class Interpolator(LeafSystem):
    def __init__(self, r_traj, rd_traj, rdd_traj, dt_traj):
        LeafSystem.__init__(self)
        self.input_t_idx = self.DeclareVectorInputPort("t", BasicVector(1)).get_index()
        self.output_r_idx = self.DeclareVectorOutputPort("r", BasicVector(3), self.get_r).get_index()
        self.output_rd_idx = self.DeclareVectorOutputPort("rd", BasicVector(3), self.get_rd).get_index()
        self.output_rdd_idx = self.DeclareVectorOutputPort("rdd", BasicVector(3), self.get_rdd).get_index()
        self.trajectory_polynomial = PiecewisePolynomial()
        t = 0.0
        for i in range(len(r_traj)-1):
            # CubicHermite assumes samples are column vectors
            r = np.reshape(r_traj[i], (3,1))
            rd = np.reshape(rd_traj[i], (3,1))
            rdd = np.reshape(rdd_traj[i], (3,1))
            dt = np.reshape(dt_traj[i+1], (3,1))
            r_next = np.reshape(r_traj[i+1], (3,1))
            rd_next = np.reshape(rd_traj[i+1], (3,1))
            rdd_next = np.reshape(rdd_traj[i+1], (3,1))
            self.trajectory_polynomial.ConcatenateInTime(
                    CubicHermite(
                        breaks=[t, t+dt],
                        samples=np.hstack([r, r_next],
                        sample_dot=[rd, rd_next])))
            t += dt
        self.trajectory = PiecewiseTrajectory(self.trajectory_polynomial)

    def get_r(self, context, output):
        t = self.EvalVectorInput(context, self.input_t_idx).get_value()
        output.SetFromVector(self.trajectory.get_position(t))

    def get_rd(self, context, output):
        t = self.EvalVectorInput(context, self.input_t_idx).get_value()
        output.SetFromVector(self.trajectory.get_velocity(t))

    def get_rdd(self, context, output):
        t = self.EvalVectorInput(context, self.input_t_idx).get_value()
        output.SetFromVector(self.trajectory.get_acceleration(t))

class ContextCache():
    def __init__(self, plant, plant_autodiff):
        self.plant = plant
        self.plant_autodiff = plant_autodiff
        self.context = plant.CreateDefaultContext()
        self.context_autodiff = plant_autodiff.CreateDefaultContext()
        self.q = None
        self.q_autodiff = None
        self.v = None
        self.v_autodiff = None

    def getPlantAndContext(self, q, v):
        assert(q.dtype == v.dtype)
        if q.dtype == np.object:
            if (np.array_equal(self.q_autodiff, q)
                    or np.array_equal(self.v_autodiff, v)):
                self.q_autodiff = q
                self.v_autodiff = v
                self.plant_autodiff.SetPositions(self.context_autodiff, self.q_autodiff)
                self.plant_autodiff.SetVelocities(self.context_autodiff, self.v_autodiff)
            return self.plant_autodiff, self.context_autodiff
        else:
            if (not np.array_equal(self.q, q)
                    or not np.array_equal(self.v, v)):
                self.q = q
                self.v = v
                self.plant.SetPositions(self.context, self.q)
                self.plant.SetVelocities(self.context, self.v)
            return self.plant, self.context

def calcTrajectory(q_init, q_final, num_knot_points, max_time, pelvis_only=False):
    N = num_knot_points
    T = max_time
    plant = MultibodyPlant(mbp_time_step)
    load_atlas(plant)
    plant_autodiff = plant.ToAutoDiffXd()
    cache = ContextCache(plant, plant_autodiff)
    upright_context = plant.CreateDefaultContext()
    set_atlas_initial_pose(plant, upright_context)
    q_nom = plant.GetPositions(upright_context)

    def get_contact_positions(q, v):
        plant_eval, context = cache.getPlantAndContext(q, v)
        lfoot_full_contact_positions = plant_eval.CalcPointsPositions(
                context, plant_eval.GetFrameByName("l_foot"),
                lfoot_full_contact_points, plant_eval.world_frame())
        rfoot_full_contact_positions = plant_eval.CalcPointsPositions(
                context, plant_eval.GetFrameByName("r_foot"),
                rfoot_full_contact_points, plant_eval.world_frame())
        return np.concatenate([lfoot_full_contact_positions, rfoot_full_contact_positions], axis=1)

    sorted_joint_position_lower_limits = np.array([entry[1].lower for entry in getSortedJointLimits(plant)])
    sorted_joint_position_upper_limits = np.array([entry[1].upper for entry in getSortedJointLimits(plant)])
    sorted_joint_velocity_limits = np.array([entry[1].velocity for entry in getSortedJointLimits(plant)])

    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(rows=N, cols=plant.num_positions(), name="q")
    v = prog.NewContinuousVariables(rows=N, cols=plant.num_velocities(), name="v")
    dt = prog.NewContinuousVariables(N, name="dt")
    r = prog.NewContinuousVariables(rows=N, cols=3, name="r")
    rd = prog.NewContinuousVariables(rows=N, cols=3, name="rd")
    rdd = prog.NewContinuousVariables(rows=N, cols=3, name="rdd")
    contact_dim = 3*num_contact_points
    # The cols are ordered as
    # [contact1_x, contact1_y, contact1_z, contact2_x, contact2_y, contact2_z, ...]
    c = prog.NewContinuousVariables(rows=N, cols=contact_dim, name="c")
    F = prog.NewContinuousVariables(rows=N, cols=contact_dim, name="F")
    tau = prog.NewContinuousVariables(rows=N, cols=contact_dim, name="tau")
    h = prog.NewContinuousVariables(rows=N, cols=3, name="h")
    hd = prog.NewContinuousVariables(rows=N, cols=3, name="hd")

    ''' Slack for the complementary constraints '''
    slack = 1e-5

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
        tauj = np.reshape(tau[k], (num_contact_points, 3))
        (prog.AddConstraint(eq(hd[k], np.sum(np.cross(cj - r[k], Fj) + tauj, axis=0)))
                .evaluator().set_description(f"Eq(7b)[{k}]"))
        ''' Eq(7c) '''
        # https://stackoverflow.com/questions/63454077/how-to-obtain-centroidal-momentum-matrix/63456202#63456202
        def eq7c(q_v_h):
            q, v, h = np.split(q_v_h, [
                plant.num_positions(),
                plant.num_positions() + plant.num_velocities()])
            plant_eval, context = cache.getPlantAndContext(q, v)
            return plant_eval.CalcSpatialMomentumInWorldAboutPoint(context, plant_eval.CalcCenterOfMassPosition(context)).rotational() - h
        (prog.AddConstraint(eq7c, lb=[0]*3, ub=[0]*3, vars=np.concatenate([q[k], v[k], h[k]]))
            .evaluator().set_description(f"Eq(7c)[{k}]"))
        ''' Eq(7h) '''
        def eq7h(q_v_r):
            q, v, r = np.split(q_v_r, [
                plant.num_positions(),
                plant.num_positions() + plant.num_velocities()])
            plant_eval, context = cache.getPlantAndContext(q, v)
            return plant_eval.CalcCenterOfMassPosition(context) - r
        # COM position has dimension 3
        (prog.AddConstraint(eq7h, lb=[0]*3, ub=[0]*3, vars=np.concatenate([q[k], v[k], r[k]]))
                .evaluator().set_description(f"Eq(7h)[{k}]"))
        ''' Eq(7i) '''
        def eq7i(q_v_ck):
            q, v, ck = np.split(q_v_ck, [
                plant.num_positions(),
                plant.num_positions() + plant.num_velocities()])
            cj = np.reshape(ck, (num_contact_points, 3))
            # print(f"q = {q}\nv={v}\nck={ck}")
            contact_positions = get_contact_positions(q, v).T
            return (contact_positions - cj).flatten()
        # np.concatenate cannot work q, cj since they have different dimensions
        (prog.AddConstraint(eq7i, lb=np.zeros(c[k].shape).flatten(), ub=np.zeros(c[k].shape).flatten(), vars=np.concatenate([q[k], v[k], c[k]]))
                .evaluator().set_description(f"Eq(7i)[{k}]"))
        ''' Eq(7j) '''
        (prog.AddBoundingBoxConstraint([-10, -10, 0]*num_contact_points, [10, 10, 10]*num_contact_points, c[k])
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
            (prog.AddLinearConstraint(le(tauj[i], np.array([0.0, 0.0, max_torque])))
                    .evaluator().set_description(f"Eq(7k)[{k}] friction torque upper limit"))
            (prog.AddLinearConstraint(ge(tauj[i], np.array([0.0, 0.0, -max_torque])))
                    .evaluator().set_description(f"Eq(7k)[{k}] friction torque lower limit"))

        ''' Assume flat ground for now... '''
        def get_contact_positions_z(q, v):
            return get_contact_positions(q, v)[2,:]
        ''' Eq(8a) '''
        def eq8a_lhs(q_v_F):
            q, v, F = np.split(q_v_F, [
                plant.num_positions(),
                plant.num_positions() + plant.num_velocities()])
            Fj = np.reshape(F, (num_contact_points, 3))
            return [Fj[:,2].dot(get_contact_positions_z(q, v))] # Constraint functions must output vectors
        (prog.AddConstraint(eq8a_lhs, lb=[-slack], ub=[slack], vars=np.concatenate([q[k], v[k], F[k]]))
                .evaluator().set_description(f"Eq(8a)[{k}]"))
        ''' Eq(8b) '''
        def eq8b_lhs(q_v_tau):
            q, v, tau = np.split(q_v_tau, [
                plant.num_positions(),
                plant.num_positions() + plant.num_velocities()])
            tauj = np.reshape(tau, (num_contact_points, 3))
            return (tauj**2).T.dot(get_contact_positions_z(q, v)) # Outputs per axis sum of torques of all contact points
        (prog.AddConstraint(eq8b_lhs, lb=[-slack]*3, ub=[slack]*3, vars=np.concatenate([q[k], v[k], tau[k]]))
                .evaluator().set_description(f"Eq(8b)[{k}]"))
        ''' Eq(8c) '''
        (prog.AddLinearConstraint(ge(Fj[:,2], 0.0))
                .evaluator().set_description(f"Eq(8c)[{k}] contact force greater than zero"))
        def eq8c_2(q_v):
            q, v = np.split(q_v, [plant.num_positions()])
            return get_contact_positions_z(q, v)
        (prog.AddConstraint(eq8c_2, lb=[0.0]*num_contact_points, ub=[float('inf')]*num_contact_points, vars=np.concatenate([q[k], v[k]]))
                .evaluator().set_description(f"Eq(8c)[{k}] z position greater than zero"))

    for k in range(1, N):
        ''' Eq(7d) '''
        def eq7d(q_qprev_v_dt):
            q, qprev, v, dt = np.split(q_qprev_v_dt, [
                plant.num_positions(),
                plant.num_positions() + plant.num_positions(),
                plant.num_positions() + plant.num_positions() + plant.num_velocities()])
            plant_eval, context = cache.getPlantAndContext(q, v)
            qd = plant_eval.MapVelocityToQDot(context, v*dt[0])
            return q - qprev - qd
        # dt[k] must be converted to an array
        (prog.AddConstraint(eq7d, lb=[0.0]*plant.num_positions(), ub=[0.0]*plant.num_positions(),
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
    ''' Eq(10) '''
    Q_q = 0.1 * np.identity(plant.num_velocities())
    Q_v = 0.2 * np.identity(plant.num_velocities())
    for k in range(N):
        def pose_error_cost(q_v_dt):
            q, v, dt = np.split(q_v_dt, [
                plant.num_positions(),
                plant.num_positions() + plant.num_velocities()])
            plant_eval, context = cache.getPlantAndContext(q, v)
            q_err = plant_eval.MapQDotToVelocity(context, q-q_nom)
            return (dt*(q_err.dot(Q_q).dot(q_err)))[0] # AddCost requires cost function to return scalar, not array
        prog.AddCost(pose_error_cost, vars=np.concatenate([q[k], v[k], [dt[k]]])) # np.concatenate requires items to have compatible shape
        prog.AddCost(dt[k]*(
                + v[k].dot(Q_v).dot(v[k])
                + rdd[k].dot(rdd[k])))

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
    (prog.AddLinearConstraint(ge(dt[1:], [1e-5]*(N-1)))
            .evaluator().set_description("min timestep"))
    (prog.AddLinearConstraint(le(dt, [1e-1]*N))
            .evaluator().set_description("max timestep"))
    '''
    Constrain unbounded variables to improve IPOPT performance
    because IPOPT is an interior point method which works poorly for unconstrained variables
    '''
    (prog.AddLinearConstraint(le(F.flatten(), np.ones(F.shape).flatten()*1e3))
            .evaluator().set_description("max F"))
    (prog.AddLinearConstraint(le(tau.flatten(), np.ones(tau.shape).flatten()*1e3))
            .evaluator().set_description("max tau"))
    (prog.AddLinearConstraint(le(beta.flatten(), np.ones(beta.shape).flatten()*1e3))
            .evaluator().set_description("max beta"))

    ''' Solve '''
    initial_guess = np.empty(prog.num_vars())
    # Guess evenly distributed dt
    prog.SetDecisionVariableValueInVector(dt, [T/N] * N, initial_guess)
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

    start_solve_time = time.time()
    solver = IpoptSolver()
    options = SolverOptions()
    options.SetOption(solver.solver_id(), "max_iter", 1000)
    print(f"Start solving...")
    result = solver.Solve(prog, initial_guess, options) # Currently takes around 30 mins
    print(f"Solve time: {time.time() - start_solve_time}s")
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
        exit(-1)
    print(f"Cost: {result.get_optimal_cost()}")
    r_sol = result.GetSolution(r)
    rd_sol = result.GetSolution(rd)
    rdd_sol = result.GetSolution(rdd)
    kt_sol = result.GetSolution(kt)

    return r_sol, rd_sol, rdd_sol, kt_sol

def main():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(mbp_time_step))
    load_atlas(plant, add_ground=True)
    plant_context = plant.CreateDefaultContext()
    set_atlas_initial_pose(plant, plant_context)

    q_init = plant.GetPositions(plant_context)
    q_final = q_init.copy()
    # q_final[4] = 0.1 # x position of pelvis
    q_final[6] = 0.90 # z position of pelvis (to make sure final pose touches ground)

    num_knot_points = 5
    max_time = 1.0

    print(f"Starting pos: {q_init}\nFinal pos: {q_final}")
    r_traj, rd_traj, rdd_traj, kt_traj = calcTrajectory(q_init, q_final, num_knot_points, max_time, pelvis_only=True)

    controller = builder.AddSystem(HumanoidController(is_wbc=True))
    controller.set_name("HumanoidController")

    ''' Connect atlas plant to controller '''
    builder.Connect(plant.get_state_output_port(), controller.GetInputPort("q_v"))
    builder.Connect(controller.GetOutputPort("tau"), plant.get_actuation_input_port())

    interpolator = builder.AddSystem(Interpolator(r_traj, rd_traj, rdd_traj, kt_traj))
    interpolator.set_name("Interpolator")
    ''' Connect interpolator to controller '''
    builder.Connect(interpolator.GetOutputPort("r"), controller.GetInputPort("r"))
    builder.Connect(interpolator.GetOutputPort("rd"), controller.GetInputPort("rd"))
    builder.Connect(interpolator.GetOutputPort("rdd"), controller.GetInputPort("rdd"))

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
