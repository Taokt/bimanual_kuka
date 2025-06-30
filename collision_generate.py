# Third Party
import torch

# cuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml


urdf_file = "/home/ji0341li/project/RoboTwin2.0/assets/embodiments/bimanual_kuka/urdf/dual_arm/dual_arm_w_grippers.urdf"
base_link = "left_arm_link_0"
ee_link = "left_arm_link_7"

# convenience function to store tensor type and device
tensor_args = TensorDeviceType()

# Generate robot configuration from  urdf path, base frame, end effector frame

robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

kin_model = CudaRobotModel(robot_cfg.kinematics)

# compute forward kinematics:
# torch random sampling might give values out of joint limits
q = torch.rand((10, kin_model.get_dof()), **(tensor_args.as_torch_dict()))
out = kin_model.get_state(q)

print("out", out)