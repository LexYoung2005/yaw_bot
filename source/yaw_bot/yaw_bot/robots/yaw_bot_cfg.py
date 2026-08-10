from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

_USD_PATH = Path(__file__).resolve().parents[4] / "assets" / "robots" / "yaw_bot" / "yaw_bot.usd"

YAW_BOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=UsdFileCfg(
        usd_path=str(_USD_PATH),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.18),
        joint_pos={
            # Keep the articulation default joint state aligned with the task-level default servo pose.
            # These values correspond to branch hip ~= 50 deg and mapped hip ~= 40 deg.
            "Body_r_1": 0.8726646259971648,
            "L_leg1_r_4": 1.5215130423567194,
            "L_leg2_r_7": 0.0,
            "Body_r_8": 0.8726646259971648,
            "R_leg1_r_9": -1.5215130423567194,
            "R_leg2_r_10": 0.0,
        },
    ),
    actuators={
        # 4个舵机：位置控制
        "hip_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "Body_r_1",
                "Body_r_8",
            ],
            stiffness=18.0,
            damping=1.2,
            effort_limit=0.45,
            velocity_limit=8.5,
        ),
        "knee_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_leg1_r_4",
                "R_leg1_r_9",
            ],
            stiffness=24.0,
            damping=1.6,
            effort_limit=0.9,
            velocity_limit=8.5,
        ),
        # 2个轮子：速度控制
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_leg2_r_7",
                "R_leg2_r_10",
            ],
            stiffness=0.0,
            damping=0.05,
            effort_limit=0.1,
            velocity_limit=125.0,
        ),
    },
)
