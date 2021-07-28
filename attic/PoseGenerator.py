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
from pydrake.all import RotationMatrix, PiecewiseTrajectory, PiecewiseQuaternionSlerp
from Atlas import getJointIndexInGeneralizedPositions
from typing import NamedTuple

class Trajectory(NamedTuple):
    frame: str
    position_offset: np.ndarray
    position_traj: PiecewiseTrajectory
    position_tolerance: float
    orientation_traj: PiecewiseQuaternionSlerp
    orientation_tolerance: float

class PoseGenerator:
    '''
    Parameters
    ----------
    trajectories : Trajectory
    '''
    def __init__(self, plant, trajectories, q_guess=None):
        self.plant = plant
        self.trajectories = trajectories
        self.q_guess = q_guess
        pass

    def get_ik(self, t):
        epsilon = 1e-2
        ik = InverseKinematics(plant=self.plant, with_joint_limits=True)

        if self.q_guess is None:
            context = self.plant.CreateDefaultContext()
            self.q_guess = self.plant.GetPositions(context)
            # This helps get the solver out of the saddle point when knee joint are locked (0.0)
            self.q_guess[getJointIndexInGeneralizedPositions(self.plant, 'l_leg_kny')] = 0.1
            self.q_guess[getJointIndexInGeneralizedPositions(self.plant, 'r_leg_kny')] = 0.1

        for trajectory in self.trajectories:
            position = trajectory.position_traj.value(t)
            ik.AddPositionConstraint(
                    frameB=self.plant.GetFrameByName(trajectory.frame),
                    p_BQ=trajectory.position_offset,
                    frameA=self.plant.world_frame(),
                    p_AQ_upper=position+trajectory.position_tolerance,
                    p_AQ_lower=position-trajectory.position_tolerance)

            if trajectory.orientation_traj is not None:
                orientation = trajectory.orientation_traj.value(t)
                ik.AddOrientationConstraint(
                        frameAbar=self.plant.world_frame(),
                        R_AbarA=RotationMatrix.Identity(),
                        frameBbar=self.plant.GetFrameByName(trajectory.frame),
                        R_BbarB=RotationMatrix(orientation),
                        theta_bound=trajectory.orientation_tolerance)
        
        result = Solve(ik.prog(), self.q_guess)
        if result.is_success():
            return result.GetSolution(ik.q())
        else:
            raise Exception("Failed to find IK solution!")
