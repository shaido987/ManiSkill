import time
from typing import Optional
import gymnasium as gym
from matplotlib.pylab import f
import numpy as np
import torch

from mani_skill.examples.benchmarking.profiling import Profiler
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
def make_eval_envs(env_id, num_envs: int, sim_backend: str, env_kwargs: dict, other_kwargs: dict, video_dir: Optional[str] = None, wrappers: list[gym.Wrapper] = []):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "cpu":
        def cpu_make_env(env_id, seed, video_dir=None, env_kwargs = dict(), other_kwargs = dict()):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=0, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True, source_type="diffusion_policy", source_desc="diffusion_policy evaluation rollout")
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk
        vector_cls = lambda x : gym.vector.AsyncVectorEnv(x, context="forkserver", **other_kwargs)
        env = vector_cls([cpu_make_env(env_id, seed, video_dir if seed == 0 else None, env_kwargs, other_kwargs) for seed in range(num_envs)])
    else:
        env = gym.make(env_id, num_envs=num_envs, sim_backend=sim_backend, reconfiguration_freq=0, **env_kwargs)
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, save_video=True, source_type="diffusion_policy", source_desc="diffusion_policy evaluation rollout", max_steps_per_video=max_episode_steps)
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env

if __name__ == "__main__":
    profiler = Profiler(output_format="stdout")
    env_kwargs = dict(render_mode="human", obs_mode="state", sim_config=dict(control_freq=20))
    num_render_envs = 1
    other_kwargs = {}
    env_id = "RoboCasaKitchen-v1"
    render_env = make_eval_envs(env_id, num_envs=num_render_envs, sim_backend="gpu", env_kwargs=env_kwargs, other_kwargs=other_kwargs)
    render_env.reset()
    num_envs = 1
    other_kwargs = {"observation_space": render_env.single_observation_space, "action_space": render_env.single_action_space}
    cpu_env = make_eval_envs(env_id, num_envs=num_envs, sim_backend="cpu", env_kwargs=env_kwargs, other_kwargs=other_kwargs)
    cpu_env.reset()
    def merge_dict(target, source):
        for key, value in source.items():
            if key not in target:
                target[key] = {}
            if isinstance(value, dict):
                merge_dict(target[key], value)
            else:
                # Convert numpy arrays to CUDA tensors
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).cuda()
                if len(target[key]) == 0:
                    target[key] = value
                else:
                    target[key] = torch.vstack([target[key], value])
    N = 100
    stime = time.time()
    for _ in range(N):
        state_dict = cpu_env.call("get_state_dict")
        # getting state dict is super slow...? with 8 envs this takes 0.005s per call, about 200 parallel steps/s
        # Merge all state dicts into a single dict
        # full_state_dict = {}



        # for item in state_dict:
        #     merge_dict(full_state_dict, item)
    print(f"Getting state into the GPU took {N / (time.time() - stime):.5f} parallel steps/s in {(time.time() - stime) / N:.5f}s per call")

    N = 100
    with profiler.profile("cpu_env.step", total_steps=N, num_envs=num_envs):
        for _ in range(N):
            action = cpu_env.action_space.sample()
            obs, reward, terminated, truncated, info = cpu_env.step(action)

    profiler.log_stats("cpu_env.step")
    # with profiler.profile("cpu_vec_render", total_steps=N, num_envs=num_render_envs):
    #     for _ in range(N):
    #         cpu_env.render()
    # profiler.log_stats("cpu_vec_render")
    cpu_env.reset()
    with profiler.profile("render", total_steps=N, num_envs=num_render_envs):
        for _ in range(N):
            action = cpu_env.action_space.sample()
            obs, reward, terminated, truncated, info = cpu_env.step(action)
            full_state_dict = {}
            if "final_info" in info:
                state_dict = [info["final_info"][x]["state_dict"] for x in range(len(info["final_info"]))]
            else:
                state_dict = info["state_dict"]
            for item in state_dict:
                merge_dict(full_state_dict, item)
            render_env.unwrapped.set_state_dict(full_state_dict)
            render_env.unwrapped.get_obs()
            # render_env.render()
    profiler.log_stats("render")
    cpu_env.close()
    render_env.close()
