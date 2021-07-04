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

    L_FOOT_HEEL_L_IDX          = 0
    L_FOOT_HEEL_R_IDX          = 1
    L_FOOT_TOE_L_IDX           = 2
    L_FOOT_TOE_R_IDX           = 3
    L_FOOT_MIDFOOT_FRONT_L_IDX = 4
    L_FOOT_MIDFOOT_FRONT_R_IDX = 5
    L_FOOT_MIDFOOT_REAR_L_IDX  = 6
    L_FOOT_MIDFOOT_REAR_R_IDX  = 7
    R_FOOT_HEEL_L_IDX          = 8
    R_FOOT_HEEL_R_IDX          = 9
    R_FOOT_TOE_L_IDX           = 10
    R_FOOT_TOE_R_IDX           = 11
    R_FOOT_MIDFOOT_FRONT_L_IDX = 12
    R_FOOT_MIDFOOT_FRONT_R_IDX = 13
    R_FOOT_MIDFOOT_REAR_L_IDX  = 14
    R_FOOT_MIDFOOT_REAR_R_IDX  = 15

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
            self.plant.GetFrameByName("l_foot_midfoot_front_l"),
            self.plant.GetFrameByName("l_foot_midfoot_front_r"),
            self.plant.GetFrameByName("l_foot_midfoot_rear_l"),
            self.plant.GetFrameByName("l_foot_midfoot_rear_r"),
            self.plant.GetFrameByName("r_foot_heel_l"),
            self.plant.GetFrameByName("r_foot_heel_r"),
            self.plant.GetFrameByName("r_foot_toe_l"),
            self.plant.GetFrameByName("r_foot_toe_r"),
            self.plant.GetFrameByName("r_foot_midfoot_front_l"),
            self.plant.GetFrameByName("r_foot_midfoot_front_r"),
            self.plant.GetFrameByName("r_foot_midfoot_rear_l"),
            self.plant.GetFrameByName("r_foot_midfoot_rear_r")]

    def set_home(self, plant, context):
        plant.SetFreeBodyPose(context, plant.GetBodyByName("pelvis"), RigidTransform([0, 0, 0.860+0.07645]))
        # Add a slight knee bend to avoid locking legs
        bend = 0.3
        # Note that at 0.2 bend, legs are actually straighter (pelvis height is higher)
        plant.GetJointByName("l_leg_hpy").set_angle(context, -bend)
        plant.GetJointByName("r_leg_hpy").set_angle(context, -bend)
        plant.GetJointByName("l_leg_kny").set_angle(context, bend)
        plant.GetJointByName("r_leg_kny").set_angle(context, bend)

    def get_stance_schedule(self):
        in_stance = np.ones((self.get_num_contacts(), self.get_num_timesteps()))
        # left foot up
        in_stance[Atlas.L_FOOT_HEEL_R_IDX, 5:15] = 0
        in_stance[Atlas.L_FOOT_HEEL_L_IDX, 5:15] = 0
        in_stance[Atlas.L_FOOT_MIDFOOT_REAR_R_IDX, 5:20] = 0
        in_stance[Atlas.L_FOOT_MIDFOOT_REAR_L_IDX, 5:20] = 0
        in_stance[Atlas.L_FOOT_MIDFOOT_FRONT_R_IDX, 5:20] = 0
        in_stance[Atlas.L_FOOT_MIDFOOT_FRONT_L_IDX, 5:20] = 0
        in_stance[Atlas.L_FOOT_TOE_R_IDX, 5:20] = 0
        in_stance[Atlas.L_FOOT_TOE_L_IDX, 5:20] = 0

        # left heel strike
        in_stance[Atlas.L_FOOT_HEEL_R_IDX, 15:] = 1
        in_stance[Atlas.L_FOOT_HEEL_L_IDX, 15:] = 1
        # left foot plant
        in_stance[Atlas.L_FOOT_MIDFOOT_REAR_R_IDX, 20:] = 1
        in_stance[Atlas.L_FOOT_MIDFOOT_REAR_L_IDX, 20:] = 1
        in_stance[Atlas.L_FOOT_MIDFOOT_FRONT_R_IDX, 20:] = 1
        in_stance[Atlas.L_FOOT_MIDFOOT_FRONT_L_IDX, 20:] = 1
        in_stance[Atlas.L_FOOT_TOE_R_IDX, 20:] = 1
        in_stance[Atlas.L_FOOT_TOE_L_IDX, 20:] = 1

        # right foot up
        in_stance[Atlas.R_FOOT_HEEL_R_IDX, 25:35] = 0
        in_stance[Atlas.R_FOOT_HEEL_L_IDX, 25:35] = 0
        in_stance[Atlas.R_FOOT_MIDFOOT_REAR_R_IDX, 25:40] = 0
        in_stance[Atlas.R_FOOT_MIDFOOT_REAR_L_IDX, 25:40] = 0
        in_stance[Atlas.R_FOOT_MIDFOOT_FRONT_R_IDX, 25:40] = 0
        in_stance[Atlas.R_FOOT_MIDFOOT_FRONT_L_IDX, 25:40] = 0
        in_stance[Atlas.R_FOOT_TOE_R_IDX, 25:40] = 0
        in_stance[Atlas.R_FOOT_TOE_R_IDX, 25:40] = 0
        ## right heel off
        #in_stance[Atlas.R_FOOT_HEEL_R_IDX, 15:25] = 0
        #in_stance[Atlas.R_FOOT_HEEL_L_IDX, 15:25] = 0
        ## right toe off
        #in_stance[Atlas.R_FOOT_MIDFOOT_REAR_R_IDX, 20:30] = 0
        #in_stance[Atlas.R_FOOT_MIDFOOT_REAR_L_IDX, 20:30] = 0
        #in_stance[Atlas.R_FOOT_MIDFOOT_FRONT_R_IDX, 20:30] = 0
        #in_stance[Atlas.R_FOOT_MIDFOOT_FRONT_L_IDX, 20:30] = 0
        #in_stance[Atlas.R_FOOT_TOE_R_IDX, 20:30] = 0
        #in_stance[Atlas.R_FOOT_TOE_L_IDX, 20:30] = 0

        # right heel strike
        in_stance[Atlas.R_FOOT_HEEL_R_IDX, 35:] = 1
        in_stance[Atlas.R_FOOT_HEEL_L_IDX, 35:] = 1
        # right foot plant
        in_stance[Atlas.R_FOOT_MIDFOOT_REAR_R_IDX, 40:] = 1
        in_stance[Atlas.R_FOOT_MIDFOOT_REAR_L_IDX, 40:] = 1
        in_stance[Atlas.R_FOOT_MIDFOOT_FRONT_R_IDX, 40:] = 1
        in_stance[Atlas.R_FOOT_MIDFOOT_FRONT_L_IDX, 40:] = 1
        in_stance[Atlas.R_FOOT_TOE_R_IDX, 40:] = 1
        in_stance[Atlas.R_FOOT_TOE_L_IDX, 40:] = 1

        return in_stance

    def get_num_timesteps(self):
        return 46

    def get_laterally_symmetric(self):
        return False

    def get_check_self_collision(self):
        return False

    def get_stride_length(self):
        return 0.35

    def get_speed(self):
        return 0.35

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
