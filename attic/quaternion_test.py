# Taken from https://stackoverflow.com/a/64778960/3177701
import numpy as np
from pydrake.all import Quaternion
from pydrake.all import MathematicalProgram, Solve, eq, le, ge, SolverOptions, SnoptSolver
from pydrake.all import Quaternion_, AutoDiffXd
import pdb

epsilon = 1e-9
quaternion_epsilon = 1e-9

def quat_multiply(q0, q1):
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                   x1*w0 + y1*z0 - z1*y0 + w1*x0,
                  -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                   x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=q0.dtype)

def apply_angular_velocity_to_quaternion(q, w_axis, w_mag, t):
    delta_q = np.hstack([np.cos(w_mag* t/2.0), w_axis*np.sin(w_mag* t/2.0)])
    return  quat_multiply(q, delta_q)

def backward_euler(q_qprev_v_dt):
    q, qprev, w_axis, w_mag, dt = np.split(q_qprev_v_dt, [
            4,
            4 + 4,
            4 + 4 + 3,
            4 + 4 + 3 + 1])
    return q - apply_angular_velocity_to_quaternion(qprev, w_axis, w_mag, dt)

N = 4
prog = MathematicalProgram()
q = prog.NewContinuousVariables(rows=N, cols=4, name='q')
w_axis = prog.NewContinuousVariables(rows=N, cols=3, name="w_axis")
w_mag = prog.NewContinuousVariables(rows=N, cols=1, name="w_mag")
# dt = [0.0] + [1.0/(N-1)] * (N-1)
dt = prog.NewContinuousVariables(rows=N, cols=1, name="dt")

for k in range(N):
    (prog.AddConstraint(lambda x: [x @ x], [1], [1], q[k])
            .evaluator().set_description(f"q[{k}] unit quaternion constraint"))
    # Impose unit length constraint on the rotation axis.
    (prog.AddConstraint(lambda x: [x @ x], [1], [1], w_axis[k])
            .evaluator().set_description(f"w_axis[{k}] unit axis constraint"))
for k in range(1, N):
    (prog.AddConstraint(lambda q_qprev_v_dt : backward_euler(q_qprev_v_dt),
            lb=[0.0]*4, ub=[0.0]*4,
            vars=np.concatenate([q[k], q[k-1], w_axis[k], w_mag[k], dt[k]]))
        .evaluator().set_description(f"q[{k}] backward euler constraint"))

(prog.AddLinearConstraint(eq(q[0], np.array([1.0, 0.0, 0.0, 0.0])))
        .evaluator().set_description("Initial orientation constraint"))
(prog.AddLinearConstraint(eq(q[-1], np.array([-0.2955511242573139, 0.25532186031279896, 0.5106437206255979, 0.7659655809383968])))
        .evaluator().set_description("Final orientation constraint"))
(prog.AddLinearConstraint(ge(dt, 0.0))
        .evaluator().set_description("Timestep greater than 0 constraint"))
(prog.AddConstraint(np.sum(dt) == 1.0)
        .evaluator().set_description("Total time constriant"))

for k in range(N):
    prog.SetInitialGuess(w_axis[k], [0, 0, 1])
    prog.SetInitialGuess(w_mag[k], [0])
    prog.SetInitialGuess(q[k], [1., 0., 0., 0.])
    prog.SetInitialGuess(dt[k], [1.0/N])
    # prog.AddCost((w_mag[k]*w_mag[k])[0])
    # prog.AddQuadraticCost(Q=np.identity(1), b=np.zeros((1,)), c = 0.0, vars=np.reshape(w_mag[k], (1,1)))
solver = SnoptSolver()
result = solver.Solve(prog)
print(result.is_success())
if not result.is_success():
    print("---------- INFEASIBLE ----------")
    print(result.GetInfeasibleConstraintNames(prog))
    print("--------------------")
print(f"Cost = {result.get_optimal_cost()}")
q_sol = result.GetSolution(q)
print(f"q_sol = {q_sol}")
w_axis_sol = result.GetSolution(w_axis)
print(f"w_axis_sol = {w_axis_sol}")
w_mag_sol = result.GetSolution(w_mag)
print(f"w_mag_sol = {w_mag_sol}")
dt_sol = result.GetSolution(dt)
print(f"dt_sol = {dt_sol}")
