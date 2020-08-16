#!/usr/bin/python3

from load_atlas import load_atlas, set_atlas_initial_pose
import numpy as np

class HumanoidPlanner(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.plant = MultibodyPlant(mbp_time_step)
        load_atlas(self.plant)
        self.upright_context = self.plant.CreateDefaultContext()
        self.q_nom = self.plant.GetPositions(self.upright_context)
