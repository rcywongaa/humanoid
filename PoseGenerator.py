'''
Given foot trajectories (
Given subset of joint positions
Produce an output pose (set of joint positions) which are
dynamically stable with closest distance to a nominal pose
Reference:
https://www.youtube.com/watch?v=E_CVq0lWfSc
'''

import numpy as np
from pydrake.all import InverseKinematics, Solve
from pydrake.all import SnoptSolver, DiagramBuilder, AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.all import RotationMatrix
from Atlas import getJointIndexInGeneralizedPositions

class PoseGenerator:
    '''
    Parameters
    ----------
    left_foot_trajectory : pydrake.PiecewiseTrajectory
    right_foot_trajectory : pydrake.PiecewiseTrajectory
    '''
    def __init__(self, plant, trajectory_dict):
        self.plant = plant
        self.trajectory_dict = trajectory_dict
        pass

    def get_ik(self, t, q_guess=None):
        epsilon = 1e-2
        # builder = DiagramBuilder()
        # _, scene_graph = AddMultibodyPlantSceneGraph(builder, self.plant)
        # diagram = builder.Build()
        # diagram_context = diagram.CreateDefaultContext()
        # plant_context = diagram.GetMutableSubsystemContext(
            # self.plant, diagram_context)
        # ik = InverseKinematics(plant=self.plant, plant_context=plant_context, with_joint_limits=False)
        ik = InverseKinematics(plant=self.plant, with_joint_limits=True)

        if q_guess is None:
            context = self.plant.CreateDefaultContext()
            q_guess = self.plant.GetPositions(context)
            q_guess[getJointIndexInGeneralizedPositions(self.plant, 'l_leg_kny')] = 0.1
            q_guess[getJointIndexInGeneralizedPositions(self.plant, 'r_leg_kny')] = 0.1

        for frame_trajectory_pair in self.trajectory_dict.items():
            position = frame_trajectory_pair[1][0].value(t)
            ik.AddPositionConstraint(
                    frameB=self.plant.GetFrameByName(frame_trajectory_pair[0]),
                    p_BQ=np.zeros(3),
                    frameA=self.plant.world_frame(),
                    p_AQ_upper=position+epsilon,
                    p_AQ_lower=position-epsilon)

            if frame_trajectory_pair[1][1] is not None:
                orientation = frame_trajectory_pair[1][1].value(t)
                ik.AddOrientationConstraint(
                        frameAbar=self.plant.world_frame(),
                        R_AbarA=RotationMatrix.Identity(),
                        frameBbar=self.plant.GetFrameByName(frame_trajectory_pair[0]),
                        R_BbarB=RotationMatrix(orientation),
                        theta_bound=0.2)
        
        # q_err = (ik.q() - q0)[7]
        # ik.prog().AddCost(np.dot(q_err, q_err))
        solver = SnoptSolver()
        result = solver.Solve(ik.prog(), q_guess)
        # result = Solve(ik.prog(), q_guess)
        print(f"Success? {result.is_success()}")
        if result.is_success():
            return result.GetSolution(ik.q())
        else:
            return None
