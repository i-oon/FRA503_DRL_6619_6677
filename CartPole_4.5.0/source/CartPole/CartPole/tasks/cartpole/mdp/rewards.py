# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def cart_pole_state_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize deviation of all cart-pole states (x, theta, x_dot, theta_dot) from zero."""
    # Extract the articulation asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Extract positions (assuming joint 0 is cart, joint 1 is pole)
    cart_pos = asset.data.joint_pos[:, 0]
    pole_pos = wrap_to_pi(asset.data.joint_pos[:, 1]) 
    
    # Extract velocities
    cart_vel = asset.data.joint_vel[:, 0]
    pole_vel = asset.data.joint_vel[:, 1]
    
    # Calculate sum of squared deviations from 0
    # Note: You can add individual scaling weights here if you want to prioritize 
    # balancing the pole over centering the cart, for example.
    penalty = (
        torch.square(cart_pos) + 
        torch.square(pole_pos) + 
        torch.square(cart_vel) + 
        torch.square(pole_vel)
    )
                    
    return penalty

def swing_up(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    cart_pos = asset.data.joint_pos[:, [0]]
    pole_joint_pos = wrap_to_pi(asset.data.joint_pos[:, [1]]) # wrap the joint positions to (-pi, pi)
    
    cart_reward = math.cos(cart_pos * math.pi / 4.8)
    pole_reward = (math.cos(pole_joint_pos) + 1) / 2.0
    reward = cart_reward * pole_reward

    reward_tensor = torch.tensor([reward], device='cuda:0')

    # print(reward_tensor)
    return reward_tensor