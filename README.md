# Humanoid Locomotion

## Controller
An Efficiently Solvable Quadratic Program for Stabilizing Dynamic Locomotion (Scott Kuindersma, Frank Permenter, and Russ Tedrake)

Roughly equivalent to [`InstantaneousQPController`](https://github.com/RobotLocomotion/drake/blob/last_sha_with_original_matlab/drake/systems/controllers/InstantaneousQPController.cpp)

### Disturbance Rejection
Rejects random forces applied to upper torso at random directions and positions with magnitude 120N for 0.1s

1. Run `python3 HumanoidController.py`
- Formulate-Solve Time: 0.04s
- 20s of simulation takes around 20mins to run

![Disturbance](resources/disturbance.gif)

### Trajectory tracking
TODO

## Planner
Whole-body Motion Planning with Centroidal Dynamics and Full Kinematics (Hongkai Dai, Andrés Valenzuela and Russ Tedrake)

Adapted from http://underactuated.mit.edu/humanoids.html#example1

### Simple walking
Simple forward walking trajectory with hard-coded contact schedule and periodicity

```
python3 Planner.py
```
- Takes around 40mins to solve
- Not solved to optimum (some constraints may be violated / may not be physically feasible)
  ```
  SNOPTA EXIT  80 -- insufficient storage allocated
  SNOPTA INFO  83 -- not enough integer storage
  ```
- Potential fix: https://stackoverflow.com/questions/68295629/solving-error-of-snopt-in-drake-how-to-fix-it
  But requires compiling from source which requires SNOPT license...
  - If someone could lend me a SNOPT license, that'd be great... :pleading_face:
- Decapitated and dismembered to improve solve time...

![Walking](resources/walking.gif)

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

### Link error on `std::filesystem` calls
Error
```
error: undefined reference to 'std::filesystem::__cxx11::path::_M_find_extension() const'
```
Use gcc 7 instead of gcc 8

### VTK problems when launching drake-visualizer
- `ModuleNotFoundError: No module named 'vtkCommonCorePython'`
- `libvtkxxx.so: No such file or directory`

In `CMakeLists.txt`, add `--define="-DUSE_SYSTEM_VTK=OFF"` after `${BAZEL_TARGETS}`

