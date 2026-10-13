from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

joint_pos_eps = 0.05


def reset_robot_controllers(env, env_ids):
    """Reset the low-level controllers of envs starting a new episode.

    `env.cfg.robot_controllers.reset`: the GPU controller resets that env's controller
    state and resamples its randomisation knobs; the CPU pool keeps its historical
    behaviour (no reset) unless built with `reset_on_episode_end=True`.
    """
    controllers = getattr(env.cfg, "robot_controllers", None)
    if controllers is not None:  # not built yet while the env itself is being built
        controllers.reset(env_ids)

@configclass
class EventsCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (0.9, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
    #         "mass_distribution_params": (-1.0, 3.0),
    #         "operation": "add",
    #     },
    # )

    # on reset
    reset_controllers = EventTerm(func=reset_robot_controllers, mode="reset")

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.1415, 3.1415)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0.0, 0.0),
    #             "yaw": (0.0, 0.0),
    #         },
    #     },
    # )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5- joint_pos_eps, 0.5 + joint_pos_eps),
            "velocity_range": (0.0, 0.0),
        },
    )
