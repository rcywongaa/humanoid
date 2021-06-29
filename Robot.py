from pydrake.all import (
        Parser
)
from abc import ABC, abstractmethod

class Robot(ABC):
    def __init__(self, plant, model_filename):
        self.plant = plant
        self.model = Parser(self.plant).AddModelFromFile(model_filename)

    def get_total_mass(self, context):
        return sum(self.plant.get_body(index).get_mass(context) for index in self.plant.GetBodyIndices(self.model))

    @abstractmethod
    def get_contact_frames(self):
        pass

    @abstractmethod
    def set_home(self, plant, context):
        pass

    def get_num_contacts(self):
        return len(self.get_contact_frames())

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

