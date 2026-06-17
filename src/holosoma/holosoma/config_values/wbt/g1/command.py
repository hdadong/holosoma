"""Whole Body Tracking command presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.1, 0.1, 0.05],
    root_ang_vel=[0.1, 0.1, 0.1],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz",
    body_names_to_track=[
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ],
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=False,
    noise_to_initial_pose=init_pose_config,
)

motion_config_w_object = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz",
)

# Fly-kick (no object). Every reset starts at motion frame 0 (deterministic) so
# the collected episodes / rewards line up with the world-model rollout. Frame 0
# is the raw fly-kick start (no interpolated default-pose prepend/append), and
# freeze-at-zero is disabled so the motion always plays forward from frame 0.
motion_config_fly_kick = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/fly_kick.npz",
    start_at_timestep_zero_prob=1.0,
    freeze_at_timestep_zero_prob=0.0,
    enable_default_pose_prepend=False,
    enable_default_pose_append=False,
)

motion_config_w_object_short_0_85_no_default_pose = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj_short_0_85.npz",
    enable_default_pose_prepend=False,
    enable_default_pose_append=False,
)

motion_config_w_object_short_0_85_freeze100_no_default_pose = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj_short_0_85_freeze100.npz",
    enable_default_pose_prepend=False,
    enable_default_pose_append=False,
)

motion_config_w_object_pre100_app100_no_default_pose = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj_pre100_app100.npz",
    enable_default_pose_prepend=False,
    enable_default_pose_append=False,
)

# LIFT3 fight1 motion with adaptive motion-frame sampling (matches the GPU0-5
# adaptive start-frame sampling). Reset init-pose noise turned OFF (all-zero
# NoiseToInitialPoseConfig) to match the GPU0-4 collected data, whose collector
# zeroes the reset pose/velocity/joint ranges (and the brax env resets exactly
# onto the motion frame with no pose noise).
motion_config_fight1_adaptive = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/motion_fight1_subject2_cut2.npz",
    use_adaptive_timesteps_sampler=True,
    noise_to_initial_pose=NoiseToInitialPoseConfig(),
)

g1_29dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
)

g1_29dof_wbt_command_fight1_adaptive = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_fight1_adaptive,
            },
        )
    },
)

g1_29dof_wbt_command_w_object = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object,
            },
        )
    },
)

g1_29dof_wbt_command_fly_kick = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_fly_kick,
            },
        )
    },
)

g1_29dof_wbt_command_w_object_short_0_85_no_default_pose = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object_short_0_85_no_default_pose,
            },
        )
    },
)

g1_29dof_wbt_command_w_object_short_0_85_freeze100_no_default_pose = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object_short_0_85_freeze100_no_default_pose,
            },
        )
    },
)

g1_29dof_wbt_command_w_object_pre100_app100_no_default_pose = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object_pre100_app100_no_default_pose,
            },
        )
    },
)

__all__ = [
    "g1_29dof_wbt_command",
    "g1_29dof_wbt_command_fight1_adaptive",
    "g1_29dof_wbt_command_fly_kick",
    "g1_29dof_wbt_command_w_object",
    "g1_29dof_wbt_command_w_object_short_0_85_no_default_pose",
    "g1_29dof_wbt_command_w_object_short_0_85_freeze100_no_default_pose",
    "g1_29dof_wbt_command_w_object_pre100_app100_no_default_pose",
]
