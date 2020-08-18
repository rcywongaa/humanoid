#!/usr/bin/python3

'''
This implements the paper
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics
by Hongkai Dai, Andrés Valenzuela and Russ Tedrake
'''

from load_atlas import load_atlas, set_atlas_initial_pose
from load_atlas import JOINT_LIMITS, lfoot_full_contact_points, rfoot_full_contact_points, FLOATING_BASE_DOF, FLOATING_BASE_QUAT_DOF, NUM_ACTUATED_DOF, TOTAL_DOF, M
from pydrake.all import eq, le, ge
from pydrake.geometry import ConnectDrakeVisualizer, SceneGraph
from pydrake.multibody.plant import ConnectContactResultsToDrakeVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from utility import calcPoseError
import numpy as np

class HumanoidPlanner(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.plant = MultibodyPlant(mbp_time_step)
        self.plant_autodiff = self.plant.ToAutoDiffXd()
        load_atlas(self.plant)
        self.upright_context = self.plant.CreateDefaultContext()
        self.q_nom = self.plant.GetPositions(self.upright_context)

        self.N = 50

        self.input_q_des_idx = self.DeclareVectorInputPort("q_des", BasicVector(self.plant.num_positions())).get_index()
        self.output_r_rd_rdd_idx = self.DeclareVectorOutputPort("r_rd_rdd", BasicVector(3*3), self.calcTrajectory).get_index()

    def calcTrajectory(self, context, output):
        prog = MathematicalProgram()
        q = prog.NewContinuousVariables(rows=N, cols=self.plant.num_positions(), name="q")
        v = prog.NewContinuousVariables(rows=N, cols=self.plant.num_velocities(), name="v")
        dt = prog.NewContinuousVariables(N, name="dt")
        r = prog.NewContinuousVariables(rows=N, cols=3, name="r")
        rd = prog.NewContinuousVariables(rows=N, cols=3, name="rd")
        rdd = prog.NewContinuousVariables(rows=N, cols=3, name="rdd")
        num_contact_points = lfoot_full_contact_points.shape[0]+rfoot_full_contact_points.shape[0]
        contact_dim = 3*num_contact_points
        # The cols are ordered as
        # [contact1_x, contact1_y, contact1_z, contact2_x, contact2_y, contact2_z, ...]
        c = prog.NewContinuousVariables(rows=N, cols=contact_dim, name="c")
        F = prog.NewContinuousVariables(rows=N, cols=contact_dim, name="F")
        tau = prog.NewContinuousVariables(rows=N, cols=contact_dim, name="tau")
        h = prog.NewContinuousVariables(rows=N, cols=3, name="h")
        hd = prog.NewContinuousVariables(rows=N, cols=3, name="hd")

        autodiff_context = self.plant_autodiff.CreateDefaultContext()
        self.plant_autodiff.SetPositions(autodiff_context, q)
        lfoot_full_contact_pos = self.plant_autodiff.CalcPointsPositions(
                autodiff_context, self.plant_autodiff.GetFrameByName("l_foot"),
                lfoot_full_contact_points, self.plant_autodiff.world_frame())
        rfoot_full_contact_pos = self.plant_autodiff.CalcPointsPositions(
                autodiff_context, self.plant_autodiff.GetFrameByName("r_foot"),
                rfoot_full_contact_points, self.plant_autodiff.world_frame())
        contact_pos = np.concatenate([lfoot_full_contact_points, rfoot_full_contact_points])

        for k in range(self.N):
            ''' Eq(7a) '''
            g = np.array([0, 0, -9.81])
            Fj = np.reshape(F[k], (num_contact_points, 3))
            prog.AddLinearConstraint(eq(M*rdd[k], np.sum(Fj, axis=0) + M*g))
            ''' Eq(7b) '''
            cj = np.reshape(c[k], (num_contact_points, 3))
            tauj = np.reshape(tau[k], (num_contact_points, 3))
            prog.AddLinearConstraint(eq(hd[k], np.sum((cj - r[k]).cross(Fj) + tauj, axis=0)))
            ''' Eq(7c) '''
            # https://stackoverflow.com/questions/63454077/how-to-obtain-centroidal-momentum-matrix/63456202#63456202
            # TODO

            ''' Eq(7h) '''
            com = self.plant_autodiff.CalcCenterOfMassPosition(autodiff_context)
            prog.AddLinearConstraint(eq(r[k], com))
            ''' Eq(7i) '''
            prog.AddLinearConstraint(eq(cj, contact_pos))
            ''' Eq(7j) '''
            # TODO
            ''' Eq(7k) '''
            # TODO
            ''' Eq(8a) '''
            # TODO
            ''' Eq(8b) '''
            # TODO
            ''' Eq(8c) '''
            # TODO
            ''' Eq(9a) '''
            # TODO
            ''' Eq(9b) '''
            # TODO

        for k in range(1, self.N):
            ''' Eq(7d) '''
            prog.AddLinearConstraint(eq(q[k] - q[k-1], v[k]*dt[k]))
            ''' Eq(7e) '''
            prog.AddLinearConstraint(eq(h[k] - h[k-1], hd[k]*dt[k]))
            ''' Eq(7f) '''
            prog.AddLinearConstraint(eq(r[k] - r[k-1], (rd[k] + rd[k-1])/2*dt[k]))
            ''' Eq(7g) '''
            prog.AddLinearConstraint(eq(rd[k] - rd[k-1], rdd[k]*dt[k]))

        ''' Eq(10) '''
        Q_q = 0.1 * np.identity(self.plant.num_velocities())
        Q_v = 0.2 * np.identity(self.plant.num_velocities())
        for k in range(self.N):
            q_err = calcPoseError(q[k], self.q_nom)
            prog.AddCost(dt[k]*(
                    q_err.dot(Q_q).dot(q_err)
                    + v[k].dot(Q_v).dot(v[k])
                    + rdd[k].dot(rdd[k])))

def main():
    pass

if __name__ == "__main__":
    main()
