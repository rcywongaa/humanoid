#!/usr/bin/python3

import numpy as np

from pydrake.all import (
        RigidTransform, Parser, PackageMap
)

from typing import NamedTuple
import time
import pdb

from Robot import Robot

'''
Floating base is attached at the pelvis link
For generalized positions, first 7 values are 4 quaternion + 3 x,y,z of floating base
For generalized velocities, first 6 values are 3 rotational velocities + 3 xd, yd, zd
Hence generalized velocities are not strictly the derivative of generalized positions
'''

class Atlas(Robot):
    class JointLimit(NamedTuple):
        effort: float
        lower: float
        upper: float
        velocity: float

    JOINT_LIMITS = {
            "back_bkx" : JointLimit(300, -0.523599, 0.523599, 12),
            "back_bky" : JointLimit(445, -0.219388, 0.538783, 9),
            "back_bkz" : JointLimit(106, -0.663225, 0.663225, 12),
            "l_arm_elx": JointLimit(112, 0, 2.35619, 12),
            "l_arm_ely": JointLimit(63,  0, 3.14159, 12),
            "l_arm_shx": JointLimit(99, -1.5708, 1.5708, 12),
            "l_arm_shz": JointLimit(87, -1.5708, 0.785398, 12),
            "l_arm_mwx": JointLimit(25, -1.7628, 1.7628, 10),
            "l_arm_uwy": JointLimit(25, -3.011, 3.011, 10),
            "l_arm_lwy": JointLimit(25, -2.9671, 2.9671, 10),
            "l_leg_akx": JointLimit(360, -0.8, 0.8, 12),
            "l_leg_aky": JointLimit(740, -1, 0.7, 12),
            "l_leg_hpx": JointLimit(530, -0.523599, 0.523599, 12),
            "l_leg_hpy": JointLimit(840, -1.61234, 0.65764, 12),
            "l_leg_hpz": JointLimit(275, -0.174358, 0.786794, 12),
            "l_leg_kny": JointLimit(890, 0,  2.35637, 12),
            "neck_ay"  : JointLimit(25, -0.602139, 1.14319, 6.28),
            "r_arm_elx": JointLimit(112, -2.35619, 0, 12),
            "r_arm_ely": JointLimit(63,  0,  3.14159, 12),
            "r_arm_shx": JointLimit(99, -1.5708, 1.5708, 12),
            "r_arm_shz": JointLimit(87, -0.785398, 1.5708, 12),
            "r_arm_mwx": JointLimit(25, -1.7628, 1.7628, 10),
            "r_arm_uwy": JointLimit(25, -3.011, 3.011, 10),
            "r_arm_lwy": JointLimit(25, -2.9671, 2.9671, 10),
            "r_leg_akx": JointLimit(360, -0.8, 0.8, 12),
            "r_leg_aky": JointLimit(740, -1, 0.7, 12),
            "r_leg_hpx": JointLimit(530, -0.523599, 0.523599, 12),
            "r_leg_hpy": JointLimit(840, -1.61234, 0.65764, 12),
            "r_leg_hpz": JointLimit(275, -0.786794, 0.174358, 12),
            "r_leg_kny": JointLimit(890, 0, 2.35637, 12)
    }

    # Taken from drake/drake-build/install/share/drake/examples/atlas/urdf/atlas_minimal_contact.urdf
    CONTACTS_PER_FRAME = {
            "l_foot": np.array([
                [-0.0876,0.066,-0.07645], # left heel
                [-0.0876,-0.0626,-0.07645], # right heel
                [0.1728,0.066,-0.07645], # left toe
                [0.1728,-0.0626,-0.07645], # right toe
                [0.086,0.066,-0.07645], # left midfoot_front
                [0.086,-0.0626,-0.07645], # right midfoot_front
                [-0.0008,0.066,-0.07645], # left midfoot_rear
                [-0.0008,-0.0626,-0.07645] # right midfoot_rear
            ]).T,
            "r_foot": np.array([
                [-0.0876,0.0626,-0.07645], # left heel
                [-0.0876,-0.066,-0.07645], # right heel
                [0.1728,0.0626,-0.07645], # left toe
                [0.1728,-0.066,-0.07645], # right toe
                [0.086,0.0626,-0.07645], # left midfoot_front
                [0.086,-0.066,-0.07645], # right midfoot_front
                [-0.0008,0.0626,-0.07645], # left midfoot_rear
                [-0.0008,-0.066,-0.07645] # right midfoot_rear
            ]).T
    }

    PELVIS_HEIGHT = 0.93845

    NUM_ACTUATED_DOF = 30
    TOTAL_DOF = 37

    # L_FOOT_HEEL_L_IDX          = 0
    # L_FOOT_HEEL_R_IDX          = 1
    # L_FOOT_TOE_L_IDX           = 2
    # L_FOOT_TOE_R_IDX           = 3
    # L_FOOT_MIDFOOT_FRONT_L_IDX = 4
    # L_FOOT_MIDFOOT_FRONT_R_IDX = 5
    # L_FOOT_MIDFOOT_REAR_L_IDX  = 6
    # L_FOOT_MIDFOOT_REAR_R_IDX  = 7
    # R_FOOT_HEEL_L_IDX          = 8
    # R_FOOT_HEEL_R_IDX          = 9
    # R_FOOT_TOE_L_IDX           = 10
    # R_FOOT_TOE_R_IDX           = 11
    # R_FOOT_MIDFOOT_FRONT_L_IDX = 12
    # R_FOOT_MIDFOOT_FRONT_R_IDX = 13
    # R_FOOT_MIDFOOT_REAR_L_IDX  = 14
    # R_FOOT_MIDFOOT_REAR_R_IDX  = 15

    L_FOOT_HEEL_L_IDX          = 0
    L_FOOT_HEEL_R_IDX          = 1
    L_FOOT_TOE_L_IDX           = 2
    L_FOOT_TOE_R_IDX           = 3
    R_FOOT_HEEL_L_IDX          = 4
    R_FOOT_HEEL_R_IDX          = 5
    R_FOOT_TOE_L_IDX           = 6
    R_FOOT_TOE_R_IDX           = 7

    L_FOOT_HEEL_IDX = [L_FOOT_HEEL_R_IDX, L_FOOT_HEEL_L_IDX]
    R_FOOT_HEEL_IDX = [R_FOOT_HEEL_R_IDX, R_FOOT_HEEL_L_IDX]
    L_FOOT_TOE_IDX = [L_FOOT_TOE_R_IDX, L_FOOT_TOE_L_IDX]
    R_FOOT_TOE_IDX = [R_FOOT_TOE_R_IDX, R_FOOT_TOE_L_IDX]

    def __init__(self, plant):
        package_map = PackageMap()
        package_map.PopulateFromFolder("robots/atlas/")
        super().__init__(plant, "robots/atlas/urdf/atlas_minimal_contact.urdf", package_map)

    def get_contact_frames(self):
        return [
            self.plant.GetFrameByName("l_foot_heel_l"),
            self.plant.GetFrameByName("l_foot_heel_r"),
            self.plant.GetFrameByName("l_foot_toe_l"),
            self.plant.GetFrameByName("l_foot_toe_r"),
            # self.plant.GetFrameByName("l_foot_midfoot_front_l"),
            # self.plant.GetFrameByName("l_foot_midfoot_front_r"),
            # self.plant.GetFrameByName("l_foot_midfoot_rear_l"),
            # self.plant.GetFrameByName("l_foot_midfoot_rear_r"),
            self.plant.GetFrameByName("r_foot_heel_l"),
            self.plant.GetFrameByName("r_foot_heel_r"),
            self.plant.GetFrameByName("r_foot_toe_l"),
            self.plant.GetFrameByName("r_foot_toe_r"),
            # self.plant.GetFrameByName("r_foot_midfoot_front_l"),
            # self.plant.GetFrameByName("r_foot_midfoot_front_r"),
            # self.plant.GetFrameByName("r_foot_midfoot_rear_l"),
            # self.plant.GetFrameByName("r_foot_midfoot_rear_r")
        ]

    def set_home(self, plant, context):
        plant.SetFreeBodyPose(context, plant.GetBodyByName("pelvis"), RigidTransform([0, 0, 0.856+0.07645]))
        # Add a slight knee bend to avoid locking legs
        hip_angle = -0.2
        knee_angle = 0.4
        ankle_angle = -0.2
        plant.GetJointByName("l_leg_hpy").set_angle(context, hip_angle)
        plant.GetJointByName("r_leg_hpy").set_angle(context, hip_angle)
        plant.GetJointByName("l_leg_kny").set_angle(context, knee_angle)
        plant.GetJointByName("r_leg_kny").set_angle(context, knee_angle)
        plant.GetJointByName("l_leg_aky").set_angle(context, ankle_angle)
        plant.GetJointByName("r_leg_aky").set_angle(context, ankle_angle)
        plant.GetJointByName("back_bky").set_angle(context, 0.05)

    def get_stance_schedule(self):
        in_stance = np.zeros((self.get_num_contacts(), self.get_num_timesteps()))

        # https://www.semanticscholar.org/paper/Algorithmic-Foundations-of-Realizing-Multi-Contact-Reher-Hereid/38c1d2cc136415076aa5d5c903202f4491c327bb/figure/2
        # 0%   Right foot heel strike (left foot heel already off)
        # 12%  Right foot toe strike
        # 24%  Left foot toe lift
        # 36%  Right foot heel lift
        # 50%  Left foot heel strike
        # 62%  Left foot toe strike
        # 74%  Right foot toe lift
        # 86%  Left foot heel lift
        # 100% Right foot heel strike

        # We start at the middle of midstance
        # equivalent to 30% of gait cycle (immediately after left toe off)

        # Right foot planted
        t = 0
        in_stance[Atlas.R_FOOT_HEEL_IDX, t:] = 1
        in_stance[Atlas.R_FOOT_TOE_IDX, t:] = 1

        # Right foot heel off
        t = 6
        in_stance[Atlas.R_FOOT_HEEL_IDX, t:] = 0

        # Left foot heel strike
        t = 20
        in_stance[Atlas.L_FOOT_HEEL_IDX, t:] = 1

        # Left foot toe strike
        t = 32
        in_stance[Atlas.L_FOOT_TOE_IDX, t:] = 1

        # Right foot toe off
        t = 44
        in_stance[Atlas.R_FOOT_TOE_IDX, t:] = 0

        return in_stance

    def get_num_timesteps(self):
        return 51

    def get_laterally_symmetric(self):
        return True

    def get_check_self_collision(self):
        return False

    def get_stride_length(self):
        return 0.7

    def get_speed(self):
        return 1.4

    def get_body_name(self):
        return "pelvis"

    def get_position_cost(self):
        q_cost = self.PositionView()([1]*self.nq)
        q_cost.pelvis_x = 0
        q_cost.pelvis_y = 0
        q_cost.pelvis_qx = 0
        q_cost.pelvis_qy = 0
        q_cost.pelvis_qz = 0
        q_cost.pelvis_qw = 0
        return q_cost

    def get_velocity_cost(self):
        v_cost = self.VelocityView()([1]*self.nv)
        v_cost.pelvis_vx = 0
        v_cost.pelvis_wx = 0
        v_cost.pelvis_wy = 0
        v_cost.pelvis_wz = 0
        return v_cost

    def get_periodic_view(self):
        q_selector = self.PositionView()([True]*self.nq)
        q_selector.pelvis_x = False
        return q_selector

    def increment_periodic_view(self, view, increment):
        view.pelvis_x += increment

    def add_periodic_constraints(self, prog, q_view, v_view):
        # Joints
        def AddAntiSymmetricPair(a, b):
            prog.AddLinearEqualityConstraint(a[0] == -b[-1])
            prog.AddLinearEqualityConstraint(a[-1] == -b[0])
        def AddSymmetricPair(a, b):
            prog.AddLinearEqualityConstraint(a[0] == b[-1])
            prog.AddLinearEqualityConstraint(a[-1] == b[0])

        AddAntiSymmetricPair(q_view.l_arm_elx, q_view.r_arm_elx)
        AddSymmetricPair(q_view.l_arm_ely, q_view.r_arm_ely)
        AddAntiSymmetricPair(q_view.l_arm_shx, q_view.r_arm_shx)
        AddAntiSymmetricPair(q_view.l_arm_shz, q_view.r_arm_shz)
        AddAntiSymmetricPair(q_view.l_arm_mwx, q_view.r_arm_mwx)
        AddSymmetricPair(q_view.l_arm_uwy, q_view.r_arm_uwy)
        AddSymmetricPair(q_view.l_arm_lwy, q_view.r_arm_lwy)

        AddAntiSymmetricPair(q_view.l_leg_akx, q_view.r_leg_akx)
        AddSymmetricPair(q_view.l_leg_aky, q_view.r_leg_aky)
        AddAntiSymmetricPair(q_view.l_leg_hpx, q_view.r_leg_hpx)
        AddSymmetricPair(q_view.l_leg_hpy, q_view.r_leg_hpy)
        AddAntiSymmetricPair(q_view.l_leg_hpz, q_view.r_leg_hpz)
        AddSymmetricPair(q_view.l_leg_kny, q_view.r_leg_kny)

        prog.AddLinearEqualityConstraint(q_view.back_bkx[0] == -q_view.back_bkx[-1])
        prog.AddLinearEqualityConstraint(q_view.back_bky[0] == q_view.back_bky[-1])
        prog.AddLinearEqualityConstraint(q_view.back_bkz[0] == -q_view.back_bkz[-1])
        prog.AddLinearEqualityConstraint(q_view.neck_ay[0] == q_view.neck_ay[-1])

        prog.AddLinearEqualityConstraint(q_view.pelvis_y[0] == -q_view.pelvis_y[-1])
        prog.AddLinearEqualityConstraint(q_view.pelvis_z[0] == q_view.pelvis_z[-1])
        # Body orientation must be in the xz plane:
        prog.AddBoundingBoxConstraint(0, 0, q_view.pelvis_qx[[0,-1]])
        prog.AddBoundingBoxConstraint(0, 0, q_view.pelvis_qz[[0,-1]])

        # Floating base velocity
        prog.AddLinearEqualityConstraint(v_view.pelvis_vx[0] == v_view.pelvis_vx[-1])
        prog.AddLinearEqualityConstraint(v_view.pelvis_vy[0] == -v_view.pelvis_vy[-1])
        prog.AddLinearEqualityConstraint(v_view.pelvis_vz[0] == v_view.pelvis_vz[-1])

    def HalfStrideToFullStride(self, a):
        b = self.PositionView()(np.copy(a))

        b.pelvis_y = -a.pelvis_y
        b.pelvis_qx = -a.pelvis_qx
        b.pelvis_qz = -a.pelvis_qz

        b.l_arm_elx = -a.r_arm_elx
        b.r_arm_elx = -a.l_arm_elx

        b.l_arm_ely = a.r_arm_ely
        b.r_arm_ely = a.l_arm_ely

        b.l_arm_shx = -a.r_arm_shx
        b.r_arm_shx = -a.l_arm_shx

        b.l_arm_shz = -a.r_arm_shz
        b.r_arm_shz = -a.l_arm_shz

        b.l_arm_mwx = -a.r_arm_mwx
        b.r_arm_mwx = -a.l_arm_mwx

        b.l_arm_uwy = a.r_arm_uwy
        b.r_arm_uwy = a.l_arm_uwy

        b.l_arm_lwy = a.r_arm_lwy
        b.r_arm_lwy = a.l_arm_lwy

        b.l_leg_akx = -a.r_leg_akx
        b.r_leg_akx = -a.l_leg_akx

        b.l_leg_aky = a.r_leg_aky
        b.r_leg_aky = a.l_leg_aky

        b.l_leg_hpx = -a.r_leg_hpx
        b.r_leg_hpx = -a.l_leg_hpx

        b.l_leg_hpy = a.r_leg_hpy
        b.r_leg_hpy = a.l_leg_hpy

        b.l_leg_hpz = -a.r_leg_hpz
        b.r_leg_hpz = -a.l_leg_hpz

        b.l_leg_kny = a.r_leg_kny
        b.r_leg_kny = a.l_leg_kny

        b.back_bkx = -a.back_bkx
        b.back_bky = a.back_bky
        b.back_bkz = -a.back_bkz
        b.neck_ay = a.neck_ay

        return b


def getAllJointIndicesInGeneralizedPositions(plant):
    for joint_limit in Atlas.JOINT_LIMITS.items():
        index = getJointIndexInGeneralizedPositions(plant, joint_limit[0])
        print(f"{joint_limit[0]}: {index}")

def getActuatorIndex(plant, joint_name):
    return int(plant.GetJointActuatorByName(joint_name + "_motor").index())

def getJointLimitsSortedByActuator(plant):
    return sorted(Atlas.JOINT_LIMITS.items(), key=lambda entry : getActuatorIndex(plant, entry[0]))

def getJointIndexInGeneralizedPositions(plant, joint_name):
    return int(plant.GetJointByName(joint_name).position_start())

def getJointIndexInGeneralizedVelocities(plant, joint_name):
    return getJointIndexInGeneralizedPositions(plant, joint_name) - 1

'''
Returns the joint limits sorted by their position in the generalized positions
'''
def getJointLimitsSortedByPosition(plant):
    return sorted(Atlas.JOINT_LIMITS.items(),
        key=lambda entry : getJointIndexInGeneralizedPositions(plant, entry[0]))

def getJointValues(plant, joint_names, context):
    ret = []
    for name in joint_names:
        ret.append(plant.GetJointByName(name).get_angle(context))
    return ret


def setJointValues(plant, joint_values, context):
    for i in range(len(joint_values)):
        plant.GetJointByIndex(i).set_angle(context, joint_values[i])
