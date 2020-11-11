import numpy as np
from pydrake.all import Quaternion
from pydrake.all import MathematicalProgram, Solve, eq, le, ge, SolverOptions, SnoptSolver
from pydrake.all import Quaternion_, AutoDiffXd
import pdb

epsilon = 1e-9
quaternion_epsilon = 1e-9
def apply_angular_velocity_to_quaternion(q, w, t):
    # This currently returns a runtime warning of division by zero
    # https://github.com/RobotLocomotion/drake/issues/10451
    norm_w = np.linalg.norm(w)
    if norm_w <= epsilon:
        return q
    norm_q = np.linalg.norm(q)
    if abs(norm_q - 1.0) > quaternion_epsilon:
        print(f"WARNING: Quaternion {q} with norm {norm_q} not normalized!")
    a = w / norm_w
    if q.dtype == AutoDiffXd:
        delta_q = Quaternion_[AutoDiffXd](np.hstack([np.cos(norm_w * t/2.0), a*np.sin(norm_w * t/2.0)]).reshape((4,1)))
        return Quaternion_[AutoDiffXd](q/norm_q).multiply(delta_q).wxyz()
    else:
        delta_q = Quaternion(np.hstack([np.cos(norm_w * t/2.0), a*np.sin(norm_w * t/2.0)]).reshape((4,1)))
        return Quaternion(q/norm_q).multiply(delta_q).wxyz()

def backward_euler(q_qprev_v_dt):
    q, qprev, v, dt = np.split(q_qprev_v_dt, [
            4,
            4+4,
            4+4+3])
    return q - apply_angular_velocity_to_quaternion(qprev, v, dt)

N = 2
prog = MathematicalProgram()
q = prog.NewContinuousVariables(rows=N, cols=4, name='q')
v = prog.NewContinuousVariables(rows=N, cols=3, name='v')
dt = prog.NewContinuousVariables(2, name='dt')
for k in range(N):
    (prog.AddConstraint(np.linalg.norm(q[k][0:4]) == 1.)
            .evaluator().set_description(f"q[{k}] unit quaternion constraint"))
for k in range(1, N):
    (prog.AddConstraint(backward_euler,
            lb=[0.0]*4, ub=[0.0]*4,
            vars=np.concatenate([q[k], q[k-1], v[k], [dt[k]]]))
        .evaluator().set_description(f"q[{k}] backward euler constraint"))

(prog.AddLinearConstraint(eq(q[0], np.array([1.0, 0.0, 0.0, 0.0])))
        .evaluator().set_description("Initial orientation constraint"))
(prog.AddLinearConstraint(eq(q[-1], np.array([-0.2955511242573139, 0.25532186031279896, 0.5106437206255979, 0.7659655809383968])))
        .evaluator().set_description("Final orientation constraint"))
(prog.AddLinearConstraint(np.sum(dt) == 1.0)
        .evaluator().set_description("Total time constraint"))

initial_guess = np.empty(prog.num_vars())
q_guess = [[1.0, 0.0, 0.0, 0.0]]*N
prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)
# v_guess = [[0., 0., 0.], [1., 2., 3.]]
# prog.SetDecisionVariableValueInVector(v, v_guess, initial_guess)
dt_guess = [0.0, 1.0]
prog.SetDecisionVariableValueInVector(dt, dt_guess, initial_guess)
solver = SnoptSolver()
result = solver.Solve(prog, initial_guess)
print(result.is_success())
if not result.is_success():
    print("---------- INFEASIBLE ----------")
    print(result.GetInfeasibleConstraintNames(prog))
    print("--------------------")
q_sol = result.GetSolution(q)
print(f"q_sol = {q_sol}")
v_sol = result.GetSolution(v)
print(f"v_sol = {v_sol}")
dt_sol = result.GetSolution(dt)
print(f"dt_sol = {dt_sol}")
pdb.set_trace()
