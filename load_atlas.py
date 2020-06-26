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

# plant is modified in place
def load_atlas(plant, is_visualize = True):
    atlas_file = FindResourceOrThrow("drake/examples/atlas/urdf/atlas_convex_hull.urdf")
    atlas = Parser(plant).AddModelFromFile(atlas_file)

    static_friction = 1.0
    green = np.array([0.5, 1.0, 0.5, 1.0])

    plant.RegisterVisualGeometry(plant.world_body(), RigidTransform(), HalfSpace(),
            "GroundVisuaGeometry", green)

    ground_friction = CoulombFriction(1.0, 1.0)
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(), HalfSpace(),
            "GroundCollisionGeometry", ground_friction)

    plant.Finalize()
    plant.set_penetration_allowance(1.0e-3)
    plant.set_stiction_tolerance(1.0e-3)

if __name__ == "__main__":
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(1.0e-3))
    load_atlas(plant)
    ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    tau = np.zeros(plant.num_actuated_dofs())
    plant.get_actuation_input_port().FixValue(plant_context, tau)

    pelvis = plant.GetBodyByName("pelvis")
    X_WP = RigidTransform(RollPitchYaw(0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.95]))
    plant.SetFreeBodyPose(plant_context, pelvis, X_WP)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(0.2)
    simulator.AdvanceTo(2.0)
