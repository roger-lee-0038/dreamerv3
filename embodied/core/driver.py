import time

import cloudpickle
import elements
import numpy as np
import portal


class Driver:

  def __init__(self, make_env_fns, parallel=True, **kwargs):
    assert len(make_env_fns) >= 1
    self.parallel = parallel
    self.kwargs = kwargs
    self.length = len(make_env_fns)
    if parallel:
      import multiprocessing as mp
      context = mp.get_context()
      self.pipes, pipes = zip(*[context.Pipe() for _ in range(self.length)])
      self.stop = context.Event()
      fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
      self.procs = [
          portal.Process(self._env_server, self.stop, i, pipe, fn, start=True)
          for i, (fn, pipe) in enumerate(zip(fns, pipes))]
      self.pipes[0].send(('act_space',))
      self.act_space = self._receive(self.pipes[0])
      self.envs = [fn() for fn in make_env_fns]
    else:
      self.envs = [fn() for fn in make_env_fns]
      self.act_space = self.envs[0].act_space
    self.callbacks = []
    self.pre_callbacks = []
    self.acts = None
    self.carry = None
    self.reset()

  def reset(self, init_policy=None):
    self.acts = {
        k: np.zeros((self.length,) + v.shape, v.dtype)
        for k, v in self.act_space.items()}
    self.acts['reset'] = np.ones(self.length, bool)
    self.carry = init_policy and init_policy(self.length)

  def close(self):
    if self.parallel:
      [proc.kill() for proc in self.procs]
    else:
      [env.close() for env in self.envs]

  def on_pre_step(self, callback):
    self.pre_callbacks.append(callback)

  def pre_steps(self, policy, preEpisodes):
    """
    Pre_steps with a number of episodes.
    preEpisodes: [
                    [(s, a, s', r), (s', a', s'', r')], 
                    ...,
                    ...,
                  ]
    no parallel
    """
    start = 0
    while start < len(preEpisodes):
      pre_episodes = preEpisodes[start:start + self.length]
      print("pre_episodes:", pre_episodes, flush=True)
      num_episodes = len(pre_episodes)
      obs_matrix = np.array([[tup[0] for tup in episode] for episode in pre_episodes])
      reward_matrix = np.array([[tup[3] for tup in episode] for episode in pre_episodes])
      action_matrix = np.array([[tup[1] for tup in episode] for episode in pre_episodes])
      print("obs_matrix:", obs_matrix.shape, "\n", obs_matrix, flush=True)
      print("reward_matrix:", reward_matrix.shape, "\n", reward_matrix, flush=True)
      print("action_matrix:", action_matrix.shape, "\n", action_matrix, flush=True)

      for le in range(obs_matrix.shape[1]):
        # For the current implementation, 
        # the following two lines are unnecessary for preload,
        # but the infomation of is_first and is_last is still useful for training.
        is_first = True if le == 0 else False # for 'reset' inside policy
        is_last = True if le == obs_matrix.shape[1] - 1 else False # for mask later
        obs = [
                self.envs[0]._obs(obs, reward, is_first=is_first, is_last=is_last) 
                for obs, reward in zip(obs_matrix[:, le], reward_matrix[:, le])
              ]
        obs = {k: np.stack([x[k] for x in obs]) for k in obs[0].keys()}
        logs = {k: v for k, v in obs.items() if k.startswith('log/')}
        obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
        print("obs:", obs, flush=True)

        acts = {
            'action': np.array(action_matrix[:, le])
            }
        print("acts:", acts, flush=True)
        #acts['reset'] = np.zeros(num_episodes, bool)
        assert all(len(x) == num_episodes for x in acts.values())
        assert all(isinstance(v, np.ndarray) for v in acts.values())
        # acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)]
        # if self.parallel:
        #   [pipe.send(('step', act)) for pipe, act in zip(self.pipes, acts)]
        #   obs = [self._receive(pipe) for pipe in self.pipes]
        # else:
        #   obs = [env.step(act) for env, act in zip(self.envs, acts)]
        # assert all(len(x) == self.length for x in obs.values()), obs
        self.carry, acts, outs = policy(self.carry, obs, known_act=acts, **self.kwargs)
        assert all(k not in acts for k in outs), (
            list(outs.keys()), list(acts.keys()))
        if obs['is_last'].any():
          mask = ~obs['is_last']
          acts = {k: self._mask(v, mask) for k, v in acts.items()}
        self.acts = {**acts, 'reset': obs['is_last'].copy()}
        trans = {**obs, **acts, **outs, **logs}
        for i in range(num_episodes):
          trn = elements.tree.map(lambda x: x[i], trans)
          [fn(trn, i, **self.kwargs) for fn in self.pre_callbacks]
        #step += len(obs['is_first'])
        #episode += obs['is_last'].sum()
        #return step, episode
        start += self.length

  def on_step(self, callback):
    self.callbacks.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    acts = self.acts # acts, dict
    assert all(len(x) == self.length for x in acts.values())
    assert all(isinstance(v, np.ndarray) for v in acts.values())
    acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)] # acts, list
    if self.parallel:
      [pipe.send(('step', act)) for pipe, act in zip(self.pipes, acts)]
      obs = [self._receive(pipe) for pipe in self.pipes]
      # write
    else:
      obs = [env.step(act) for env, act in zip(self.envs, acts)] # obs, list
    obs = {k: np.stack([x[k] for x in obs]) for k in obs[0].keys()}
    logs = {k: v for k, v in obs.items() if k.startswith('log/')}
    obs = {k: v for k, v in obs.items() if not k.startswith('log/')} # obs, dict
    assert all(len(x) == self.length for x in obs.values()), obs
    self.carry, acts, outs = policy(self.carry, obs, **self.kwargs) # acts, dict
    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()} # acts, dict
    self.acts = {**acts, 'reset': obs['is_last'].copy()}
    trans = {**obs, **acts, **outs, **logs}
    for i in range(self.length):
      trn = elements.tree.map(lambda x: x[i], trans)
      [fn(trn, i, **self.kwargs) for fn in self.callbacks]
    step += len(obs['is_first'])
    episode += obs['is_last'].sum()
    return step, episode

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)

  def _receive(self, pipe):
    try:
      msg, arg = pipe.recv()
      if msg == 'error':
        raise RuntimeError(arg)
      assert msg == 'result'
      return arg
    except Exception:
      print('Terminating workers due to an exception.')
      [proc.kill() for proc in self.procs]
      raise

  @staticmethod
  def _env_server(stop, envid, pipe, ctor):
    try:
      ctor = cloudpickle.loads(ctor)
      env = ctor()
      while not stop.is_set():
        if not pipe.poll(0.1):
          time.sleep(0.1)
          continue
        try:
          msg, *args = pipe.recv()
        except EOFError:
          return
        if msg == 'step':
          assert len(args) == 1
          act = args[0]
          obs = env.step(act)
          pipe.send(('result', obs))
        elif msg == 'obs_space':
          assert len(args) == 0
          pipe.send(('result', env.obs_space))
        elif msg == 'act_space':
          assert len(args) == 0
          pipe.send(('result', env.act_space))
        else:
          raise ValueError(f'Invalid message {msg}')
    except ConnectionResetError:
      print('Connection to driver lost')
    except Exception as e:
      pipe.send(('error', e))
      raise
    finally:
      try:
        env.close()
      except Exception:
        pass
      pipe.close()
