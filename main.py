import torch
import math

import random
import sys
import time
sys.path.append('../kinova-swig-wrapper')
sys.path.append('../jaco-simulation')

import ipdb
import kinova

import numpy as np

from dm_control import suite
import jaco
import pyglet
import math

import inspect
import cv2
from dm_control.mujoco.wrapper.mjbindings import mjlib


LOCAL_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}
suite._DOMAINS = {**suite._DOMAINS, **LOCAL_DOMAINS}
env = suite.load(domain_name="jaco", task_name="basic")

width = 640
height = 480
fullwidth = width * 2

action_spec = env.action_spec()
time_step = env.reset()

action = np.zeros([9])
time_step = env.step(action)

def move_target_to_hand():
  env.physics.named.model.geom_pos['target'] = env.physics.named.data.xpos['jaco_link_hand']

def move_mocap_to_hand():
  env.physics.named.data.mocap_pos['endpoint'] = env.physics.named.data.xpos['jaco_link_hand']



# def zero_mocap_offset():
#   env.physics.named.model.eq_data['weld'].fill(0)

cv2.namedWindow('arm', cv2.WINDOW_NORMAL)
cv2.resizeWindow('arm', fullwidth, height)

def render():
    pixel1 = env.physics.render(height, width, camera_id=1)
    pixel2 = env.physics.render(height, width, camera_id=2)
    pixel = np.concatenate([pixel1, pixel2], 1)
    cv2.imshow('arm', pixel)
    cv2.waitKey(10)

def get_jaco_angles():
    pos = kinova.get_angular_position()
    angles = [a for a in pos.Actuators] + [a for a in pos.Fingers]
    return angles

def move_mujoco_to_real():
    angles = get_jaco_angles()
    env.physics.named.data.qpos[:9] = real_to_sim(angles)

gears = env.physics.model.actuator_gear[:, 0]

# where the robot has to be (in kinova coordinates)
# to be at zero in mujoco
zero_offset = np.array([-180, 270, 90, 180, 180, -90, 0, 0, 0])

# correct for the different physical directions of a +theta
# movement between mujoco
directions = np.array([-1, 1, -1, -1, -1, -1, 1, 1, 1])

# correct for the degrees -> radians shift going from arm
# to mujoco
scales = np.array([math.pi / 180] * 6 + [0.78 / 6800] * 3)

def real_to_sim(angles):
    return (angles - zero_offset) * directions * scales

kinova.start()

# controls = np.array(env.physics.named.data.qpos[:6])
print('starting position: ')
print(env.physics.named.data.qpos[:9])
# controls[0] = 0

move_mujoco_to_real()
while True:
    for i in range(100):
        angles = get_jaco_angles()

        env.step(real_to_sim(angles) * gears)
        print(env.physics.named.data.qpos[:9])

        # sometimes mujoco flips out and resets the sim's angles to all zeros
        if np.allclose(np.zeros(9), env.physics.named.data.qpos[:9]):
            move_mujoco_to_real()

        render()

    # import ipdb; ipdb.set_trace()
cv2.destroyAllWindows()
