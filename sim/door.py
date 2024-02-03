from sim.simulator import BaseEnv
import numpy as np
import os

class DoorEnv(BaseEnv):
    def __init__(self, 
                 seed = None,
                 default_renderer_size=256, 
                 default_camera_name="backview", 
                 device_id=-1, 
                 depth=False,
                 config = None,
                 ):
        #获取当前文件的路径
        current_path = str(os.path.dirname(os.path.abspath(__file__)))
        super().__init__(current_path+"/door.xml", seed, default_renderer_size, default_camera_name, device_id, depth,config)
    ''' #只看门是否打开的wstep
    def wstep(self,action):
        prev = self.sim.data.get_joint_qpos('Door_hinge')
        self.sim.data.ctrl[:] = action
        self.sim.step()
        curr = self.sim.data.get_joint_qpos('Door_hinge')
        reward = curr - prev
        done = curr >= 1.5 or self.curr_time >= 20
        return reward,done
    '''
    def wstep(self, action):
        if self.config is not None:
            ratio = self.config["ratio"]
            multiply = self.config["multiply"]
        else:
            ratio = 0
            multiply = 1

        #改为限速
        self.sim.data.qvel[:] = np.clip(self.sim.data.qvel[:],-10,10)
        self.sim.data.qpos[:] = np.clip(self.sim.data.qpos[:],-3,3)
        self.sim.forward()

        prev_door = self.sim.data.get_joint_qpos('Door_hinge')
        prev_robot = self.sim.data.get_body_xpos('robot0_base')[0]
        self.sim.data.ctrl[:] = action
        self.sim.step()
        curr_door = self.sim.data.get_joint_qpos('Door_hinge')
        curr_robot = self.sim.data.get_body_xpos('robot0_base')[0]
        
        done = curr_door >= 1.5 or self.curr_time >= 10 # 门打开/时间到10秒
        reward_door = np.clip(curr_door - prev_door,-1.5,1.5)
        reward_robot = np.clip(curr_robot - prev_robot,-3,3)
        reward = multiply*(ratio * reward_door + reward_robot)/(1.5*ratio+3) #奖励归一化+成倍数
        return reward, done
    
    def wreset(self):
        self.door_open = False
        #随机执行一步动作
        action = np.random.uniform(-1., 1., self.action_size)
        action_min = self.inner_action_space[:, 0]
        action_max = self.inner_action_space[:, 1]
        action_range = action_max - action_min
        self.sim.data.ctrl[:] = (action + 1) * (action_range / 2) + action_min
        self.sim.step()
