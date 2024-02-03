#本文件为环境包装
import numpy as np

#将动作归一化+数据类型转换
class StandardWrapper:
    def __init__(self, env, dtype=np.float32):
        self._env = env
        self._dtype = dtype
        #将动作空间归一化
        self._env.action_space = np.ones_like(self._env.action_space).astype(self._dtype)
        self._env.action_space[:, 0] = -1.

    def step(self, action):
        action_min = self._env.inner_action_space[:, 0].astype(self._dtype)
        action_max = self._env.inner_action_space[:, 1].astype(self._dtype)
        action_range = action_max - action_min
        original_action = ((action + 1) * (action_range / 2) + action_min).astype(self._dtype)
        return self._env.step(original_action)
    
    def __getattr__(self, name):
        return getattr(self._env, name)
    
#设置动作重复次数
class RepeatWrapper: #设置动作重复次数
    def __init__(self, env, repeat):
        self._env = env
        self._repeat = repeat
        self._env.time_step = self._env.time_step * repeat
    
    def step(self, action):
        reward = 0
        for _ in range(self._repeat):
            obs, r, done= self._env.step(action)
            reward += r
            if done:
                break
        return obs, reward, done
    
    def __getattr__(self, name):
        return getattr(self._env, name)
    
# frame_stack wrapper：最好包在最里层！！！
class StackWrapper:
    def __init__(self, env, stack, human=True): #human参数决定返回什么东西
        self._env = env
        assert stack >= 1
        self._stack = stack
        self._frames = []
        self.human = human
        # 修改观测空间大小
        self._env.observation_size = (stack, self._env.observation_size)
        self._env.human_observation_size = (stack*self._env.human_observation_size[0],)+self._env.human_observation_size[1:]#修改人类观测空间大小
    
    def step(self, action):
        obs, reward, done = self._env.step(action)
        if self.human:
            self._frames.append(self._env.render()) #仅支持默认摄像机渲染,返回为H*W*C的数组
        else:
            self._frames.append(obs)
        if len(self._frames) > self._stack:
            self._frames.pop(0)
        if self._stack>1:
            return np.concatenate(self._frames, axis = 0), reward, done
        else:
            return self._frames[-1], reward, done
    
    def reset(self):
        self._frames = []
        if self.human:
            self._env.reset()
            obs = self._env.render()
        else:
            obs = self._env.reset()
        for _ in range(self._stack):
            self._frames.append(obs)
        if self._stack>1:
            return np.concatenate(self._frames, axis = 0)#直接返回一个观察值
        else:
            return self._frames[-1]
    
    def __getattr__(self, name):
        return getattr(self._env, name)
    

def Wrap(env, action_repeat=5, obs_stack=8, dtype=np.float32,human=True):
    env = StackWrapper(env, obs_stack, human)
    env = StandardWrapper(env)
    env = RepeatWrapper(env, action_repeat)
    return env