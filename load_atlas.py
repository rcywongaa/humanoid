#!/usr/bin/python3

import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.all import MultibodyPlant
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import ConnectDrakeVisualizer, SceneGraph, HalfSpace, Box
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.systems.analysis import Simulator
from pydrake.math import RollPitchYaw
from collections import namedtuple

'''
Floating base is attached at the pelvis link
'''

JointLimit = namedtuple("JointLimit", ["effort", "lower", "upper"])

JOINT_LIMITS = {
        "back_bkx" : JointLimit(300, -0.523599, 0.523599),
        "back_bky" : JointLimit(445, -0.219388, 0.538783),
        "back_bkz" : JointLimit(106, -0.663225, 0.663225),
        "l_arm_elx": JointLimit(112, 0, 2.35619),
        "l_arm_ely": JointLimit(63,  0, 3.14159),
        "l_arm_shx": JointLimit(99, -1.5708, 1.5708),
        "l_arm_shz": JointLimit(87, -1.5708, 0.785398),
        "l_arm_mwx": JointLimit(25, -1.7628, 1.7628),
        "l_arm_uwy": JointLimit(25, -3.011, 3.011),
        "l_arm_lwy": JointLimit(25, -2.9671, 2.9671),
        "l_leg_akx": JointLimit(360, -0.8, 0.8),
        "l_leg_aky": JointLimit(740, -1, 0.7),
        "l_leg_hpx": JointLimit(530, -0.523599, 0.523599),
        "l_leg_hpy": JointLimit(840, -1.61234, 0.65764),
        "l_leg_hpz": JointLimit(275, -0.174358, 0.786794),
        "l_leg_kny": JointLimit(890, 0,  2.35637),
        "neck_ay"  : JointLimit(25, -0.602139, 1.14319),
        "r_arm_elx": JointLimit(112, -2.35619, 0),
        "r_arm_ely": JointLimit(63,  0,  3.14159),
        "r_arm_shx": JointLimit(99, -1.5708, 1.5708),
        "r_arm_shz": JointLimit(87, -0.785398, 1.5708),
        "r_arm_mwx": JointLimit(25, -1.7628, 1.7628),
        "r_arm_uwy": JointLimit(25, -3.011, 3.011),
        "r_arm_lwy": JointLimit(25, -2.9671, 2.9671),
        "r_leg_akx": JointLimit(360, -0.8, 0.8),
        "r_leg_aky": JointLimit(740, -1, 0.7),
        "r_leg_hpx": JointLimit(530, -0.523599, 0.523599),
        "r_leg_hpy": JointLimit(840, -1.61234, 0.65764),
        "r_leg_hpz": JointLimit(275, -0.786794, 0.174358),
        "r_leg_kny": JointLimit(890, 0, 2.35637)
}

# Taken from drake/drake-build/install/share/drake/examples/atlas/urdf/atlas_minimal_contact.urdf
lfoot_full_contact_points = np.array([
    [-0.0876,0.066,-0.07645], # left heel
    [-0.0876,-0.0626,-0.07645], # right heel
    [0.1728,0.066,-0.07645], # left toe
    [0.1728,-0.0626,-0.07645], # right toe
    [0.086,0.066,-0.07645], # left midfoot_front
    [0.086,-0.0626,-0.07645], # right midfoot_front
    [-0.0008,0.066,-0.07645], # left midfoot_rear
    [-0.0008,-0.0626,-0.07645] # right midfoot_rear
]).T

rfoot_full_contact_points = np.array([
    [-0.0876,0.0626,-0.07645], # left heel
    [-0.0876,-0.066,-0.07645], # right heel
    [0.1728,0.0626,-0.07645], # left toe
    [0.1728,-0.066,-0.07645], # right toe
    [0.086,0.0626,-0.07645], # left midfoot_front
    [0.086,-0.066,-0.07645], # right midfoot_front
    [-0.0008,0.0626,-0.07645], # left midfoot_rear
    [-0.0008,-0.066,-0.07645] # right midfoot_rear
]).T

FLOATING_BASE_DOF = 6
FLOATING_BASE_QUAT_DOF = 7 # Start index of actuated joints in generalized positions
NUM_ACTUATED_DOF = 30
TOTAL_DOF = FLOATING_BASE_DOF + NUM_ACTUATED_DOF

# plant is modified in place
def load_atlas(plant, add_ground=False):
    atlas_file = FindResourceOrThrow("drake/examples/atlas/urdf/atlas_minimal_contact.urdf")
    # atlas_file = FindResourceOrThrow("drake/examples/atlas/urdf/atlas_convex_hull.urdf")
    atlas = Parser(plant).AddModelFromFile(atlas_file)

    if add_ground:
        static_friction = 1.0
        green = np.array([0.5, 1.0, 0.5, 1.0])

        # plant.RegisterVisualGeometry(plant.world_body(), RigidTransform(), HalfSpace(),
                # "GroundVisuaGeometry", green)

        ground_friction = CoulombFriction(1.0, 1.0)
        plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(), HalfSpace(),
                "GroundCollisionGeometry", ground_friction)

    plant.Finalize()
    plant.set_penetration_allowance(1.0e-3)
    plant.set_stiction_tolerance(1.0e-3)

def set_atlas_initial_pose(plant, plant_context):
    pelvis = plant.GetBodyByName("pelvis")
    X_WP = RigidTransform(RollPitchYaw(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.94]))
    plant.SetFreeBodyPose(plant_context, pelvis, X_WP)

def set_null_input(plant, plant_context):
    tau = np.zeros(plant.num_actuated_dofs())
    plant.get_actuation_input_port().FixValue(plant_context, tau)


if __name__ == "__main__":
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(1.0e-3))
    load_atlas(plant, add_ground=True)
    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    set_atlas_initial_pose(plant, plant_context)
    set_null_input(plant, plant_context)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(0.2)
    simulator.AdvanceTo(2.0)
