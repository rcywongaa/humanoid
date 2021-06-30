import numpy as np
from pydrake.all import (
        Parser, RigidTransform
)

from Robot import Robot

class LittleDog(Robot):
    def __init__(self, plant, gait="walking_trot"):
        super().__init__(plant, "robots/littledog/LittleDog.urdf")

        # setup gait
        self.is_laterally_symmetric = False
        self.check_self_collision = False
        if gait == 'running_trot':
            self.N = 21
            self.in_stance = np.zeros((4, self.N))
            self.in_stance[1, 3:17] = 1
            self.in_stance[2, 3:17] = 1
            self.speed = 0.9
            self.stride_length = .55
            self.is_laterally_symmetric = True
        elif gait == 'walking_trot':
            self.N = 21
            self.in_stance = np.zeros((4, self.N))
            self.in_stance[0, :11] = 1
            self.in_stance[1, 8:self.N] = 1
            self.in_stance[2, 8:self.N] = 1
            self.in_stance[3, :11] = 1
            self.speed = 0.4
            self.stride_length = .25
            self.is_laterally_symmetric = True
        elif gait == 'rotary_gallop':
            self.N = 41
            self.in_stance = np.zeros((4, self.N))
            self.in_stance[0, 7:19] = 1
            self.in_stance[1, 3:15] = 1
            self.in_stance[2, 24:35] = 1
            self.in_stance[3, 26:38] = 1
            self.speed = 1
            self.stride_length = .65
            self.check_self_collision = True
        elif gait == 'bound':
            self.N = 41
            self.in_stance = np.zeros((4, self.N))
            self.in_stance[0, 6:18] = 1
            self.in_stance[1, 6:18] = 1
            self.in_stance[2, 21:32] = 1
            self.in_stance[3, 21:32] = 1
            self.speed = 1.2
            self.stride_length = .55
            self.check_self_collision = True
        else:
            raise RuntimeError('Unknown gait.')

    def get_contact_frames(self): 
        return [
            self.plant.GetFrameByName('front_left_foot_center'),
            self.plant.GetFrameByName('front_right_foot_center'),
            self.plant.GetFrameByName('back_left_foot_center'),
            self.plant.GetFrameByName('back_right_foot_center')]

    def set_home(self, plant, context):
        hip_roll = .1;
        hip_pitch = 1;
        knee = 1.55;
        plant.GetJointByName("front_right_hip_roll").set_angle(context, -hip_roll)
        plant.GetJointByName("front_right_hip_pitch").set_angle(context, hip_pitch)
        plant.GetJointByName("front_right_knee").set_angle(context, -knee)
        plant.GetJointByName("front_left_hip_roll").set_angle(context, hip_roll)
        plant.GetJointByName("front_left_hip_pitch").set_angle(context, hip_pitch)
        plant.GetJointByName("front_left_knee").set_angle(context, -knee)
        plant.GetJointByName("back_right_hip_roll").set_angle(context, -hip_roll)
        plant.GetJointByName("back_right_hip_pitch").set_angle(context, -hip_pitch)
        plant.GetJointByName("back_right_knee").set_angle(context, knee)
        plant.GetJointByName("back_left_hip_roll").set_angle(context, hip_roll)
        plant.GetJointByName("back_left_hip_pitch").set_angle(context, -hip_pitch)
        plant.GetJointByName("back_left_knee").set_angle(context, knee)
        plant.SetFreeBodyPose(context, plant.GetBodyByName("body"), RigidTransform([0, 0, 0.146]))

    def get_stance_schedule(self):
        return self.in_stance

    def get_num_timesteps(self):
        return self.N

    def get_laterally_symmetric(self):
        return self.is_laterally_symmetric

    def get_check_self_collision(self):
        return self.check_self_collision

    def get_stride_length(self):
        return self.stride_length

    def get_speed(self):
        return self.speed

    def get_body_name(self):
        return "body"
