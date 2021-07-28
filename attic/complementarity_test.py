import numpy as np
from pydrake.all import Quaternion
from pydrake.all import MathematicalProgram, Solve, eq, le, ge, SolverOptions, SnoptSolver
import pdb

prog = MathematicalProgram()
x = prog.NewContinuousVariables(rows=1, name='x')
y = prog.NewContinuousVariables(rows=1, name='y')
slack = prog.NewContinuousVariables(rows=1, name="slack")

prog.AddConstraint(eq(x*y, slack))
prog.AddLinearConstraint(ge(x, 0))
prog.AddLinearConstraint(ge(y, 0))
prog.AddLinearConstraint(le(x, 1))
prog.AddLinearConstraint(le(y, 1))
prog.AddCost(1e5*(slack**2)[0])
prog.AddCost(-(2*x[0]+y[0]))

solver = SnoptSolver()
result = solver.Solve(prog)
print(f"Success: {result.is_success()}, x = {result.GetSolution(x)}, y = {result.GetSolution(y)}, slack = {result.GetSolution(slack)}")
