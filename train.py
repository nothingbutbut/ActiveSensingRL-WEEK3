import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from sim.door import DoorEnv
from sim.wrapper import Wrap
from video import recorder

import os
from pathlib import Path
import hydra
import numpy as np
import torch
import utils
from replay_buffer import ReplayBufferStorage, make_replay_loader
from collections import defaultdict

import wandb
from logger import Logger
import datetime

torch.backends.cudnn.benchmark = True

ENVS = {
    'door': DoorEnv,
}

def make_agent(obs_shape, action_shape, cfg):
    cfg.obs_shape = obs_shape
    cfg.action_shape = action_shape
    return hydra.utils.instantiate(cfg)

def wrap_step(obs, action, reward, discount, done):
    return {
        'observation': obs,
        'action': action,
        'reward': reward,
        'discount': discount,
        'last': done,
    }

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        self.agent = make_agent(self.train_env.human_observation_size,
                                (self.train_env.action_size,), #注意要将其先包在元组里
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        wandb.login(key="831dd2e7df54e5d34423a2494fb33008677e2b6a")
        now = datetime.datetime.now()
        year = now.year
        month = now.month
        date = now.day
        time = now.strftime("%H:%M:%S") 
        name="Test"
        time=f"{year}-{month}-{date} {time}"
        wandb.init(
            # set the wandb project where this run will be logged
            project="DRQV2-Open the door IV",
            entity="brucekang",
            name=name+time,
            tags=["test"],
            config={
                "ratio": self.cfg.ratio,
                "render_size": self.cfg.render_size,
                "action_repeat": self.cfg.action_repeat,
                "multiply": self.cfg.multiply,
            }
        )

        # create envs
        self.train_env = Wrap(ENVS[self.cfg.task_name](self.cfg.seed, self.cfg.render_size, self.cfg.cam_name, self.cfg.device_id, config = {"ratio":self.cfg.ratio,"multiply":self.cfg.multiply}),
                              action_repeat=self.cfg.action_repeat,
                              obs_stack=self.cfg.frame_stack
                            )
        self.eval_env = Wrap(ENVS[self.cfg.task_name](self.cfg.seed, self.cfg.render_size, self.cfg.cam_name, self.cfg.device_id, config = {"ratio":self.cfg.ratio,"multiply":self.cfg.multiply}),
                              action_repeat=self.cfg.action_repeat,
                              obs_stack=self.cfg.frame_stack
                            )
        
        # create replay buffer
        data_specs = {
            'observation': np.zeros(self.train_env.human_observation_size, np.float32),
            'action': np.zeros(self.train_env.action_size, np.float32),
            'reward': np.zeros(1, np.float32),
            'discount': np.zeros(1, np.float32),
        }
        # 示例step没有done,不需要存储这一部分
        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer',minimal_episode_len=self.cfg.nstep+1)

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = recorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = recorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            obs = self.eval_env.reset()
            self.video_recorder.init(f'{self.global_frame}.mp4',size=self.train_env.default_renderer_size, fps = int(1/self.train_env.time_step), enabled=(episode == 0))
            done = False
            while not done:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, #记得先交换维度顺序
                                            self.global_step,
                                            eval_mode=True)
                obs,reward,done = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env.render()[-3:,::]) #只能取最后三个部分(3*256*256)
                total_reward += reward
                step += 1

            episode += 1
            self.video_recorder.release()

        wandb.log(
            {
                'eval_episode_reward': total_reward / episode,
                'eval_episode_length': step * self.cfg.action_repeat / episode,
                'eval_episode': self.global_episode,
                'eval_step': self.global_step,
             }, step=self.global_frame
        )
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
        
    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        obs = self.train_env.reset()
        time_step = wrap_step(obs, np.zeros(self.train_env.action_size, np.float32), np.zeros(1, np.float32), np.zeros(1, np.float32), False)
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(f'{self.global_frame}.mp4',size=self.train_env.default_renderer_size, fps = int(1/self.train_env.time_step))
        metrics = None
        while train_until_step(self.global_step):
            if time_step["last"]:
                #print("current length of replay buffer: ", len(self.replay_storage))
                #print("current frame count:", self.global_frame)
                self._global_episode += 1
                self.train_video_recorder.release()
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    wandb.log({
                        'train_fps': episode_frame / elapsed_time,
                        'train_total_time': total_time,
                        'train_episode_reward': episode_reward,
                        'train_episode_length': episode_frame,
                        'train_episode': self.global_episode,
                        'train_step': self.global_step,
                        'buffer_size': len(self.replay_storage)
                    }, step=self.global_frame)
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                obs = self.train_env.reset()
                time_step = wrap_step(obs, np.zeros(self.train_env.action_size, np.float32), np.zeros(1, np.float32), np.zeros(1, np.float32), False)
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(f'{self.global_frame}.mp4',size=self.train_env.default_renderer_size, fps = int(1/self.train_env.time_step))
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                wandb.log({'eval_total_time': self.timer.total_time()}, step=self.global_frame)
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step["observation"], #记得先交换维度顺序
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step) and len(self.replay_storage) >= self.cfg.min_replay_buffer_size:
                #print("current length of replay buffer: ", len(self.replay_storage))
                #print("current frame count:", self.global_frame)
                metrics = self.agent.update(self.replay_iter, self.global_step)
                wandb.log(metrics, step=self.global_frame)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            obs,reward,done = self.train_env.step(action)
            time_step = wrap_step(obs, action, reward, self.cfg.discount, done)
            episode_reward += time_step["reward"]
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step["observation"][-3:,::])
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()