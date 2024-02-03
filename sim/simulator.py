#本文件包含了基本的环境类，以及一个缩减的renderer类
import cv2
import numpy as np
from robosuite.utils.binding_utils import MjRenderContextOffscreen
from robosuite.utils.binding_utils import MjSim
import mujoco
from PIL import Image

# 缩减的renderer:不再能保存视频，只能进行渲染
class Renderer:
    def __init__(
        self,
        sim,
        cam_name="upview",
        width=256,
        height=256,
        depth=False,
        device_id=-1,
        seed = None,
    )->None:
        self._render_context = MjRenderContextOffscreen(sim, device_id=device_id)
        sim.add_render_context(self._render_context)
        self._width = width
        self._height = height
        self._depth = depth
        self._cam_id = sim.model.camera_name2id(cam_name)
        self.length = 0
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            self.sim.seed(seed)

    def render(self, depth=False):
        self._render_context.render(
            width=self._width,
            height=self._height,
            camera_id=self._cam_id,
        )
        img = self._render_context.read_pixels(
            self._width,
            self._height,
            depth=depth,
        )
        #增加的步骤：还原色彩
        #img = Image.frombytes("RGB", (self._width,self._height),img)
        #img = np.asarray(img) #返回一个numpy数组
        img = img.transpose(2,0,1) #转置为CHW
        return img
    
class BaseEnv:
    def __init__(self,
                 model_path,
                 seed,
                 default_renderer_size=256,
                 default_camera_name="backview",
                 device_id=-1,
                 depth = False,
                 config = None,
                 )->None:
        #set up the environment
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.sim = MjSim(self.model)

        # renderers setting
        self.default_renderer_size = default_renderer_size
        self.human_observation_size = (3, default_renderer_size, default_renderer_size) #渲染图大小：考虑stackwrapper
        self.inner_human_observation_size = (3, default_renderer_size, default_renderer_size) #渲染图大小：不考虑stackwrapper
        self.default_camera_name = default_camera_name
        self.depth = depth
        self.device_id = device_id
        self.renderers = {}
        self.renderers['default'] = Renderer(self.sim, default_camera_name, default_renderer_size, default_renderer_size,depth=depth, device_id=device_id)

        # info setting
        self.time_step = self.sim.model.opt.timestep #时间步长 要考虑repeatwrapper
        self.inner_time_step = self.sim.model.opt.timestep #时间步长 不考虑repeatwrapper
        self.action_space = self.sim.model.actuator_ctrlrange # 动作空间 要考虑standardwrapper
        self.inner_action_space = self.sim.model.actuator_ctrlrange # 动作空间 不考虑standardwrapper
        self.action_size = len(self.action_space)
        self.observation_size = len(self.sim.data.qpos[:]) #观测空间 要考虑stackwrapper
        self.inner_observation_size = len(self.sim.data.qpos[:]) #观测空间 不考虑stackwrapper
        self.num_steps = 0
        self.curr_time = 0
        self.config = config

        # seed setting
        self.seed(seed)

    def get_action_size(self):
        return self.action_size
        
    def get_observation_size(self):
        return self.observation_size
    
    def get_obs(self):
        return self.sim.data.qpos[:] #返回位置
        
    def wstep(self,action): #被包裹的step函数
        pass
        
    def step(self,action):
        reward,done = self.wstep(action) 
        obs = self.get_obs()
        self.num_steps += 1
        self.curr_time += self.inner_time_step
        '''
        # 稳定性检查：检查所有的qpos和qvel是否超出范围
        for x in self.sim.data.qpos:
            if x > 1e3 or x < -1e3:
                done = True
                break
        for x in self.sim.data.qvel:
            if x > 1e3 or x < -1e3:
                done = True
                break
        
        if done:
            self.reset()
        '''
        return obs,reward,done
        
    def wreset(self): #被包裹的reset函数
        pass

    def reset(self):
        self.sim.reset()
        self.num_steps = 0
        self.curr_time = 0
        self.wreset()
        return self.get_obs()
        
    def render(self, renderer='default'):
        return self.renderers[renderer].render(self.depth)
        
    def add_renderer(self, cam_name, renderer_name, width=None, height=None, depth=None):
        if width is None:
            width = self.default_renderer_size
        if height is None:
            height = self.default_renderer_size
        if depth is None:
            depth = self.depth
        self.renderers[renderer_name] = Renderer(self.sim, cam_name, width, height, depth, self.device_id)

    def seed(self,seed):
        if seed is not None:
            np.random.seed(seed)
            #self.sim.seed(seed)
    
    def get_human_obsevation_size(self): #返回人类观测空间大小
        return self.human_obsevation_size