from pydrake.all import (
        Parser, JointIndex
)
from pydrake.common.containers import namedview
from abc import ABC, abstractmethod
import pdb

def MakeNamedViewPositions(mbp, view_name):
    names = [None]*mbp.num_positions()
    for ind in range(mbp.num_joints()): 
        joint = mbp.get_joint(JointIndex(ind))
        # TODO: Handle planar joints, etc.
        if joint.num_positions() < 1:
            continue
        assert(joint.num_positions() == 1)
        names[joint.position_start()] = joint.name()
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        start = body.floating_positions_start()
        body_name = body.name()
        names[start] = body_name+'_qw'
        names[start+1] = body_name+'_qx'
        names[start+2] = body_name+'_qy'
        names[start+3] = body_name+'_qz'
        names[start+4] = body_name+'_x'
        names[start+5] = body_name+'_y'
        names[start+6] = body_name+'_z'
    return namedview(view_name, names)

def MakeNamedViewVelocities(mbp, view_name):
    names = [None]*mbp.num_velocities()
    for ind in range(mbp.num_joints()): 
        joint = mbp.get_joint(JointIndex(ind))
        if joint.num_positions() < 1:
            continue
        # TODO: Handle planar joints, etc.
        assert(joint.num_velocities() == 1)
        names[joint.velocity_start()] = joint.name()
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        start = body.floating_velocities_start() - mbp.num_positions()
        body_name = body.name()
        names[start] = body_name+'_wx'
        names[start+1] = body_name+'_wy'
        names[start+2] = body_name+'_wz'
        names[start+3] = body_name+'_vx'
        names[start+4] = body_name+'_vy'
        names[start+5] = body_name+'_vz'
    return namedview(view_name, names)

class Robot(ABC):
    def __init__(self, plant, model_filename, package_map=None):
        self.plant = plant
        parser = Parser(self.plant)
        if package_map is not None:
            parser.package_map().AddMap(package_map)
        self.model = parser.AddModelFromFile(model_filename)
        self.plant.Finalize()
        self.nq = self.plant.num_positions()
        self.nv = self.plant.num_velocities()

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

    @abstractmethod
    def get_position_cost(self):
        pass

    @abstractmethod
    def get_velocity_cost(self):
        pass

    @abstractmethod
    def get_periodic_view(self):
        pass

    @abstractmethod
    def increment_periodic_view(self, view, increment):
        pass

    def get_total_mass(self, context):
        return sum(self.plant.get_body(index).get_mass(context) for index in self.plant.GetBodyIndices(self.model))

    def get_num_contacts(self):
        return len(self.get_contact_frames())

    def PositionView(self):
        return MakeNamedViewPositions(self.plant, "Positions")

    def VelocityView(self):
        return MakeNamedViewVelocities(self.plant, "Velocities")
