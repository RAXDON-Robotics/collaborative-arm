// bindings/wrapper.cpp

#include <pybind11/functional.h> // 如果以后有 callback
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 支持 std::vector
// 如果需要支持 numpy arrays 或 buffer，就引入 numpy.h 或 buffer.h

#include "y1_sdk/y1_sdk_interface.h"

namespace py = pybind11;

PYBIND11_MODULE(y1_sdk, m) {
  m.doc() = "Python binding for Y1SDKInterface";

  // 绑定枚举 ControlMode
  py::enum_<raxdon::y1_controller::Y1SDKInterface::ControlMode>(m, "ControlMode")
      .value("GRAVITY_COMPENSATION",
             raxdon::y1_controller::Y1SDKInterface::GRAVITY_COMPENSATION)
      .value("RT_JOINT_POSITION",
             raxdon::y1_controller::Y1SDKInterface::RT_JOINT_POSITION)
      .value("NRT_JOINT_POSITION",
             raxdon::y1_controller::Y1SDKInterface::NRT_JOINT_POSITION)
      .export_values();

  // 绑定类 Y1SDKInterface
  py::class_<raxdon::y1_controller::Y1SDKInterface>(m, "Y1SDKInterface")
      // 构造函数
      .def(py::init<const std::string &, const std::string &, int, bool>(),
           py::arg("can_id"), py::arg("urdf_path"), py::arg("arm_end_type"),
           py::arg("enable_arm"))
      // 析构函数自动处理

      // 方法
      .def("Init", &raxdon::y1_controller::Y1SDKInterface::Init,
           "Initialize the SDK interface. Returns true if success.")
      .def("GetJointNames",
           &raxdon::y1_controller::Y1SDKInterface::GetJointNames,
           "Returns the joint names (6 or 7 including gripper).")
      .def("GetRotorTemperature",
           &raxdon::y1_controller::Y1SDKInterface::GetRotorTemperature,
           "Returns rotor (coil) temperature for all joints.")
      .def("GetJointErrorCode",
           &raxdon::y1_controller::Y1SDKInterface::GetJointErrorCode,
           "Returns error codes for all joints.")
      .def("GetMotorCurrent",
           &raxdon::y1_controller::Y1SDKInterface::GetMotorCurrent,
           "Returns motor current for all joints.")
      .def("GetJointPosition",
           &raxdon::y1_controller::Y1SDKInterface::GetJointPosition,
           "Joint positions.")
      .def("GetJointVelocity",
           &raxdon::y1_controller::Y1SDKInterface::GetJointVelocity,
           "Joint velocities.")
      .def("GetJointEffort",
           &raxdon::y1_controller::Y1SDKInterface::GetJointEffort,
           "Joint torques / efforts.")
      .def("GetArmEndPose",
           &raxdon::y1_controller::Y1SDKInterface::GetArmEndPose,
           "End pose of the arm: [x, y, z, roll, pitch, yaw]")

      .def("SetArmControlMode",
           &raxdon::y1_controller::Y1SDKInterface::SetArmControlMode,
           "Set control mode", py::arg("mode"))

      // 控制函数
      .def("SetArmJointPosition",
           (void (raxdon::y1_controller::Y1SDKInterface::*)(
               const std::array<double, 6> &,
               int))&raxdon::y1_controller::Y1SDKInterface::SetArmJointPosition,
           "Set joint positions by std::array<double,6> with optional velocity "
           "ratio",
           py::arg("arm_joint_position"), py::arg("velocity_ratio") = 5)

      .def(
          "SetFollowerArmJointPosition",
          (void (raxdon::y1_controller::Y1SDKInterface::*)(
              const std::vector<double>
                  &))&raxdon::y1_controller::Y1SDKInterface::SetArmJointPosition,
          "Set joint positions by vector<double>",
          py::arg("arm_joint_position"))

      .def("SetArmEndPose",
           &raxdon::y1_controller::Y1SDKInterface::SetArmEndPose,
           "Set end pose [x,y,z,roll,pitch,yaw] and return calculated joint "
           "positions from inverse kinematics, with optional velocity ratio",
           py::arg("arm_end_pose"), py::arg("velocity_ratio") = 5)

      .def("SetGripperStroke",
           &raxdon::y1_controller::Y1SDKInterface::SetGripperStroke,
           "Set gripper stroke in mm", py::arg("gripper_stroke"),
           py::arg("velocity_ratio") = 5)

      .def("SetEnableArm", &raxdon::y1_controller::Y1SDKInterface::SetEnableArm,
           "Enable or disable arm motors", py::arg("enable_flag"))

      .def("SaveJ6ZeroPosition",
           &raxdon::y1_controller::Y1SDKInterface::SaveJ6ZeroPosition,
           "Save zero position for J6 joint when changing end flange.");

  // 结束 module
}
