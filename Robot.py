from pydrake.all import (
        Parser
)
from abc import ABC, abstractmethod
import pdb

class Robot(ABC):
    def __init__(self, plant, model_filename, package_map=None):
        self.plant = plant
        parser = Parser(self.plant)
        if package_map is not None:
            parser.package_map().AddMap(package_map)
        self.model = parser.AddModelFromFile(model_filename)

    @abstractmethod
    def get_contact_frames(self):
        pass

    @abstractmethod
    def set_home(self, plant, context):
        pass

    @abstractmethod
    def get_stance_schedule(self):
        pass

    @abstractmethod
    def get_num_timesteps(self):
        pass

    @abstractmethod
    def get_laterally_symmetric(self):
        pass

    @abstractmethod
    def get_check_self_collision(self):
        pass

    @abstractmethod
    def get_stride_length(self):
        pass

    @abstractmethod
    def get_speed(self):
        pass

    @abstractmethod
    def get_body_name(self):
        pass

    def get_total_mass(self, context):
        return sum(self.plant.get_body(index).get_mass(context) for index in self.plant.GetBodyIndices(self.model))

    def get_num_contacts(self):
        return len(self.get_contact_frames())

