from sim.door import DoorEnv
from sim.wrapper import Wrap
import numpy as np
from tqdm import trange
from video import recorder
import os
import time

size = 256

# 创建环境
se = 1706844605 #int(time.time())
print("seed:",se)
env = DoorEnv(seed=se, default_renderer_size=size, default_camera_name="backview", config={"ratio": 1.0, "multiply": 1.0})

# 增加包装
env = Wrap(env,action_repeat=20,obs_stack=8,human=True)

# 打印环境的一些属性
print('Action space:', env.get_action_size())
print('Observation space:', env.get_observation_size())
print("human sees :",env.human_observation_size)

#设置视频保存
print("time for a single step:",env.time_step)
cwd = os.getcwd()
backview = recorder(cwd)
firstview = recorder(cwd)

backview.init("demo_backview.mp4",size=size,fps=int(1/env.time_step),enabled=True)
firstview.init("demo_firstview.mp4",size=size,fps=int(1/env.time_step),enabled=True)

# 执行一些动作
env.reset()
env.add_renderer("birdview","firstview")
total= 0
maximal = -20
door_pos = -20
robot_pos = -20
print("original qpos",env.sim.data.get_joint_qpos('Door_hinge'))
print("original xpos",env.sim.data.get_body_xpos('robot0_base'))
for i in trange(int(10/env.time_step)):
    action = np.random.uniform(-1, 1, size=env.get_action_size()) #采用随机动作
    pos, reward,done = env.step(action)
    if env.sim.data.get_joint_qpos('Door_hinge')>door_pos:
        door_pos = env.sim.data.get_joint_qpos('Door_hinge')
    if env.sim.data.get_body_xpos('robot0_base')[0]>robot_pos:
        robot_pos = env.sim.data.get_body_xpos('robot0_base')[0]
    if env.sim.data.get_body_xpos('robot0_base')[0]>3 or env.sim.data.get_body_xpos('robot0_base')[0]<-2:
        print("out of range")
        print(env.sim.data.get_body_xpos('robot0_base'))
    if done:
        print("done!")
    total+=reward
    if reward > maximal:
        maximal = reward
    shape = pos.shape
    backview.record(pos[-3:,::])
    firstview.record(env.render(renderer="firstview"))

print("obs shape:",shape)
print("total reward:",total)
print("maximal reward:",maximal)
print("MAXIMAL robot pos:",robot_pos)
print("MAXIMAL door pos:",door_pos)
print("current robot pos:",env.sim.data.get_body_xpos('robot0_base')[0])
print("current door pos:",env.sim.data.get_joint_qpos('Door_hinge'))
backview.release()
firstview.release()