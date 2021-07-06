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

    def get_position_cost(self):
        q_cost = self.PositionView()([1]*self.nq)
        q_cost.body_x = 0
        q_cost.body_y = 0
        q_cost.body_qx = 0
        q_cost.body_qy = 0
        q_cost.body_qz = 0
        q_cost.body_qw = 0
        q_cost.front_left_hip_roll = 5
        q_cost.front_right_hip_roll = 5
        q_cost.back_left_hip_roll = 5
        q_cost.back_right_hip_roll = 5
        return q_cost

    def get_velocity_cost(self):
        v_cost = self.VelocityView()([1]*self.nv)
        v_cost.body_vx = 0
        v_cost.body_wx = 0
        v_cost.body_wy = 0
        v_cost.body_wz = 0
        return v_cost

    def get_periodic_view(self):
        q_selector = self.PositionView()([True]*self.nq)
        q_selector.body_x = False
        return q_selector

    def increment_periodic_view(self, view, increment):
        view.body_x += increment

    def add_periodic_constraints(self, prog, q_view, v_view):
        # Joints
        def AddAntiSymmetricPair(a, b):
            prog.AddLinearEqualityConstraint(a[0] == -b[-1])
            prog.AddLinearEqualityConstraint(a[-1] == -b[0])
        def AddSymmetricPair(a, b):
            prog.AddLinearEqualityConstraint(a[0] == b[-1])
            prog.AddLinearEqualityConstraint(a[-1] == b[0])

        AddAntiSymmetricPair(q_view.front_left_hip_roll,
                             q_view.front_right_hip_roll)
        AddSymmetricPair(q_view.front_left_hip_pitch,
                         q_view.front_right_hip_pitch)
        AddSymmetricPair(q_view.front_left_knee, q_view.front_right_knee)
        AddAntiSymmetricPair(q_view.back_left_hip_roll,
                             q_view.back_right_hip_roll)
        AddSymmetricPair(q_view.back_left_hip_pitch,
                         q_view.back_right_hip_pitch)
        AddSymmetricPair(q_view.back_left_knee, q_view.back_right_knee)
        prog.AddLinearEqualityConstraint(q_view.body_y[0] == -q_view.body_y[-1])
        prog.AddLinearEqualityConstraint(q_view.body_z[0] == q_view.body_z[-1])
        # Body orientation must be in the xz plane:
        prog.AddBoundingBoxConstraint(0, 0, q_view.body_qx[[0,-1]])
        prog.AddBoundingBoxConstraint(0, 0, q_view.body_qz[[0,-1]])

        # Floating base velocity
        prog.AddLinearEqualityConstraint(v_view.body_vx[0] == v_view.body_vx[-1])
        prog.AddLinearEqualityConstraint(v_view.body_vy[0] == -v_view.body_vy[-1])
        prog.AddLinearEqualityConstraint(v_view.body_vz[0] == v_view.body_vz[-1])

    def HalfStrideToFullStride(self, a):
        b = self.PositionView()(np.copy(a))

        b.body_y = -a.body_y
        # Mirror quaternion so that roll=-roll, yaw=-yaw
        b.body_qx = -a.body_qx
        b.body_qz = -a.body_qz

        b.front_left_hip_roll = -a.front_right_hip_roll
        b.front_right_hip_roll = -a.front_left_hip_roll
        b.back_left_hip_roll = -a.back_right_hip_roll
        b.back_right_hip_roll = -a.back_left_hip_roll

        b.front_left_hip_pitch = a.front_right_hip_pitch
        b.front_right_hip_pitch = a.front_left_hip_pitch
        b.back_left_hip_pitch = a.back_right_hip_pitch
        b.back_right_hip_pitch = a.back_left_hip_pitch

        b.front_left_knee = a.front_right_knee
        b.front_right_knee = a.front_left_knee
        b.back_left_knee = a.back_right_knee
        b.back_right_knee = a.back_left_knee

        return b

