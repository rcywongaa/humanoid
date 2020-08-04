# Humanoid Locomotion

## Setup

### Install drake
<https://drake.mit.edu/python_bindings.html#python-bindings-binary>

## Run
1. Launch `drake-visualizer`
   ```
   ./drake-visualizer
   ```
1. Run `python3 balance.py`

## Troubleshooting

### Out of memory when building drake
In `CMakeLists.txt`, add `--jobs 4` after `${BAZEL_TARGETS}`
```
ExternalProject_Add(drake_cxx_python
  SOURCE_DIR "${PROJECT_SOURCE_DIR}"
  CONFIGURE_COMMAND :
  BUILD_COMMAND
    ${BAZEL_ENV}
    "${Bazel_EXECUTABLE}"
    ${BAZEL_STARTUP_ARGS}
    build
    ${BAZEL_ARGS}
    ${BAZEL_TARGETS}
    --jobs 4
  BUILD_IN_SOURCE ON
  BUILD_ALWAYS ON
  INSTALL_COMMAND
    ${BAZEL_ENV}
    "${Bazel_EXECUTABLE}"
    ${BAZEL_STARTUP_ARGS}
    run
    ${BAZEL_ARGS}
    ${BAZEL_TARGETS}
    --
    ${BAZEL_TARGETS_ARGS}
  USES_TERMINAL_BUILD ON
  USES_TERMINAL_INSTALL ON
)
```

### `ModuleNotFoundError: No module named 'vtkCommonCorePython'` when launching drake-visualizer
In `tools/workspace/drake_visualizer/repository.bzl` set `USE_SYSTEM_VTK=OFF`

## Resources

### Simulation resources

#### Thormang (2019)
- <https://github.com/thor-mang>
- <https://github.com/ROBOTIS-GIT/ROBOTIS-THORMANG-OPC>

#### R5 (2017)
- <https://github.com/osrf-migration/srcsim-wiki>
- <https://github.com/osrf/srcsim>
- <https://github.com/osrf/srcsim_docker/tree/master/docker>

#### Atlas (2016)
- <https://github.com/osrf/drcsim>
- <https://github.com/osrf-migration/drcsim-wiki>

### Whole body controllers
Responsible for converting foot and body / center of mass trajectories into actual joint torques
- <https://github.com/poftwaresatent/whole_body_control>
- <https://github.com/poftwaresatent/stanford_wbc>
- <https://bitbucket.org/ihmcrobotics/ihmc_ros>
- [ControlIt!](http://sites.utexas.edu/hcrl/files/2016/01/ijhr-2015.pdf)

### Mathematical tools
#### General robot modeling and optimal control
- <https://github.com/RobotLocomotion/drake>
#### Kinematics & Dynamics
- <https://github.com/ANYbotics/kindr>
#### Trajectory optimization
Responsible for creating foot and body / center of mass trajectories given start point and end goal
- <https://github.com/ethz-adrl/towr>

### Other tools
#### Motion / Task controlling
- <https://github.com/leggedrobotics/free_gait>

#### Visualization
- <http://wiki.ros.org/xpp>
- <https://github.com/RobotLocomotion/director>
