#!/usr/bin/python3

'''
This implements the paper
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics
by Hongkai Dai, Andrés Valenzuela and Russ Tedrake
'''

from load_atlas import load_atlas, set_atlas_initial_pose
from load_atlas import getSortedJointLimits, getActuatorIndex, getActuatorIndices, getJointValues
from load_atlas import JOINT_LIMITS, lfoot_full_contact_points, rfoot_full_contact_points, FLOATING_BASE_DOF, FLOATING_BASE_QUAT_DOF, NUM_ACTUATED_DOF, TOTAL_DOF, M
from pydrake.all import eq, le, ge, PiecewisePolynomial, PiecewiseTrajectory
from pydrake.geometry import ConnectDrakeVisualizer, SceneGraph
from pydrake.multibody.plant import ConnectContactResultsToDrakeVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.all import IpoptSolver
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.systems.framework import BasicVector, LeafSystem
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

def calcTrajectory(q_init, q_final, num_knot_points, max_time, pelvis_only=False):
    N = num_knot_points
    T = max_time
    plant = MultibodyPlant(mbp_time_step)
    load_atlas(plant)
    plant_autodiff = plant.ToAutoDiffXd()
    upright_context = plant.CreateDefaultContext()
    q_nom = plant.GetPositions(upright_context)

    def get_contact_positions(q, v):
        plant_eval = plant_autodiff if q.dtype == np.object else plant
        context = plant_eval.CreateDefaultContext()
        plant_eval.SetPositions(context, q)
        plant_eval.SetVelocities(context, v)
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
        # TODO

        ''' Eq(7h) '''
        def eq7h(q_v_r):
            plant_eval = plant_autodiff if q_v_r.dtype == np.object else plant
            q, v, r = np.split(q_v_r, [
                plant.num_positions(),
                plant.num_positions() + plant.num_velocities()])
            context = plant_eval.CreateDefaultContext()
            plant_eval.SetPositions(context, q)
            plant_eval.SetVelocities(context, v)
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
        ''' We don't constrain the contact point positions for now... '''

        ''' Eq(7k) '''
        ''' Constrain admissible posture '''
        (prog.AddLinearConstraint(le(q[k, FLOATING_BASE_QUAT_DOF:], sorted_joint_position_upper_limits))
                .evaluator().set_description(f"Eq(7k)[{k}] joint position upper limit"))
        (prog.AddLinearConstraint(ge(q[k, FLOATING_BASE_QUAT_DOF:], sorted_joint_position_lower_limits))
                .evaluator().set_description(f"Eq(7k)[{k}] joint position lower limit"))
        ''' Constrain velocities '''
        (prog.AddLinearConstraint(le(v[k, FLOATING_BASE_DOF:], sorted_joint_velocity_limits))
                .evaluator().set_description(f"Eq(7k)[{k}] joint velocity upper limit"))
        (prog.AddLinearConstraint(ge(v[k, FLOATING_BASE_DOF:], -sorted_joint_velocity_limits))
                .evaluator().set_description(f"Eq(7k)[{k}] joint velocity lower limit"))
        ''' Constrain forces within friction cone '''
        beta_k = np.reshape(beta[k], (num_contact_points, N_d))
        for i in range(num_contact_points):
            beta_v = beta_k[i].dot(friction_cone_components[:,i,:])
            (prog.AddLinearConstraint(eq(Fj[i], beta_v))
                    .evaluator().set_description(f"Eq(7k)[{k}] friction cone constraint"))
        ''' Constrain beta positive '''
        for b in beta.flat:
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
        # prog.AddConstraint(eq8a_lhs, lb=[0.0], ub=[0.0], vars=np.concatenate([q[k], v[k], F[k]]))
        ''' Eq(8b) '''
        def eq8b_lhs(q_v_tau):
            q, v, tau = np.split(q_v_tau, [
                plant.num_positions(),
                plant.num_positions() + plant.num_velocities()])
            tauj = np.reshape(tau, (num_contact_points, 3))
            return (tauj**2).T.dot(get_contact_positions_z(q, v)) # Outputs per axis sum of torques of all contact points
        # prog.AddConstraint(eq8b_lhs, lb=[0.0]*3, ub=[0.0]*3, vars=np.concatenate([q[k], v[k], tau[k]]))
        ''' Eq(8c) '''
        # prog.AddLinearConstraint(ge(Fj[:,2], 0.0))
        # TODO: Fix infeasible constraint
        # prog.AddConstraint(get_contact_positions_z, lb=[0.0]*num_contact_points, ub=[float('inf')]*num_contact_points, vars=q[k])

    for k in range(1, N):
        ''' Eq(7d) '''
        def eq7d(q_qprev_v_dt):
            plant_eval = plant_autodiff if q_qprev_v_dt.dtype == np.object else plant
            q, qprev, v, dt = np.split(q_qprev_v_dt, [
                plant.num_positions(),
                plant.num_positions() + plant.num_positions(),
                plant.num_positions() + plant.num_positions() + plant.num_velocities()])
            context = plant_eval.CreateDefaultContext()
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
            # prog.AddConstraint(Fj[i,2] * (cj[i] - cj_prev[i]).dot(np.array([1.0, 0.0, 0.0])) == 0.0)
            ''' Eq(9b) '''
            # prog.AddConstraint(Fj[i,2] * (cj[i] - cj_prev[i]).dot(np.array([0.0, 1.0, 0.0])) == 0.0)
    ''' Eq(10) '''
    Q_q = 0.1 * np.identity(plant.num_velocities())
    Q_v = 0.2 * np.identity(plant.num_velocities())
    for k in range(N):
        def pose_error_cost(q_dt):
            plant_eval = plant_autodiff if q_dt.dtype == np.object else plant
            q, dt = np.split(q_dt, [plant_eval.num_positions()])
            q_err = plant_eval.MapQDotToVelocity(plant_eval.CreateDefaultContext(), q-q_nom)
            return (dt*(q_err.dot(Q_q).dot(q_err)))[0] # AddCost requires cost function to return scalar, not array
        prog.AddCost(pose_error_cost, vars=np.concatenate([q[k], [dt[k]]]))
        prog.AddCost(dt[k]*(
                + v[k].dot(Q_v).dot(v[k])
                + rdd[k].dot(rdd[k])))

    ''' Additional constraints not explicitly stated '''
    ''' Constrain initial pose '''
    (prog.AddLinearConstraint(eq(q[0], q_init))
            .evaluator().set_description("initial pose"))
    ''' Constrain initial velocity '''
    # prog.AddLinearConstraint(eq(v[0], 0.0))
    ''' Constrain final pose '''
    if pelvis_only:
        (prog.AddLinearConstraint(eq(q[-1, 4:7], q_final[4:7]))
                .evaluator().set_description("final pose"))
    else:
        (prog.AddLinearConstraint(eq(q[-1], q_final))
                .evaluator().set_description("final pose"))
    ''' Constrain final velocity '''
    # prog.AddLinearConstraint(eq(v[0], 0.0))
    ''' Constrain time taken '''
    (prog.AddLinearConstraint(np.sum(dt) <= T)
            .evaluator().set_description("max time"))
    ''' Constrain time step '''
    (prog.AddLinearConstraint(ge(dt, [1e-5]*N))
            .evaluator().set_description("min timestep"))
    (prog.AddLinearConstraint(le(dt, [1e-1]*N))
            .evaluator().set_description("max timestep"))
    '''
    Constrain F to improve IPOPT performance
    because IPOPT is an interior point method which works poorly for unconstrained variables
    '''
    (prog.AddLinearConstraint(le(F.flatten(), np.ones(F.shape).flatten()*10000))
            .evaluator().set_description("max F"))

    ''' Solve '''
    initial_guess = np.empty(prog.num_vars())
    # Guess evenly distributed dt
    prog.SetDecisionVariableValueInVector(dt, [T/N] * N, initial_guess)
    # Guess q to avoid initializing with invalid quaternion
    prog.SetDecisionVariableValueInVector(q, [q_init] * N, initial_guess)
    # TODO: Linear interpolate q, taking care of quaternion
    # TODO: Guess v based on interpolation

    start_solve_time = time.time()
    print(f"Start solving...")
    solver = IpoptSolver()
    result = solver.Solve(prog, initial_guess) # Currently takes around 15 mins
    print(f"Solve time: {time.time() - start_solve_time}s")
    if not result.is_success():
        print(f"FAILED")
        print(result.GetInfeasibleConstraintNames(prog))
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

    num_knot_points = 100
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
