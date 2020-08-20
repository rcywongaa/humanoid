from pydrake.all import Quaternion, Quaternion_, Expression, AutoDiffXd
import numpy as np
from load_atlas import FLOATING_BASE_DOF, FLOATING_BASE_QUAT_DOF, NUM_ACTUATED_DOF, TOTAL_DOF
import pdb

def normalize(q):
    return q / np.linalg.norm(q)

# From http://www.nt.ntnu.no/users/skoge/prost/proceedings/ecc-2013/data/papers/0927.pdf
def calcAngularError(q_source, q_target):
    try:
        quat_err = Quaternion(normalize(q_source)).multiply(Quaternion(normalize(q_target)).inverse())
        if quat_err.w() < 0:
            return -quat_err.xyz()
        else:
            return quat_err.xyz()
    except:
        pdb.set_trace()

# FIXME
def calcAngularError_Expression(q_source, q_target):
    try:
        quat_err = Quaternion_[Expression](normalize(q_source)).multiply(Quaternion_[Expression](normalize(q_target)).inverse())
        return quat_err.w() * quat_err.xyz()
    except:
        pdb.set_trace()


def calcPoseError(source, target):
    assert(target.size == source.size)
    # Make sure pose is expressed in generalized positions (quaternion base)
    assert(target.size == NUM_ACTUATED_DOF + FLOATING_BASE_QUAT_DOF)

    error = np.zeros(target.shape[0]-1)
    error[0:3] = calcAngularError(source[0:4], target[0:4])
    error[3:] = (source - target)[4:]
    return error

# FIXME
def calcPoseError_Expression(source, target):
    assert(target.size == source.size)
    # Make sure pose is expressed in generalized positions (quaternion base)
    assert(target.size == NUM_ACTUATED_DOF + FLOATING_BASE_QUAT_DOF)

    error = np.zeros(target.shape[0]-1, dtype=Expression)
    error[0:3] = calcAngularError_Expression(source[0:4], target[0:4])
    error[3:] = (source - target)[4:]
    return error

