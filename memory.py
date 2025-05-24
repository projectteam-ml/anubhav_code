"""
Source: https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg.py
"""
import numpy as np



class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def clear(self):
        self.start = 0
        self.length = 0
        self.data[:] = 0


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, observation_shape, action_shape, next_actions=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        # print("action shape:",action_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        # self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.terminals = RingBuffer(limit, shape=(1,))

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        # batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        batch_idxs = random_machine.randint(0, self.nb_entries, size=batch_size)

        '''states_batch = array_min2d(self.states.get_batch(batch_idxs))
        actions_batch = array_min2d(self.actions.get_batch(batch_idxs))
        rewards_batch = array_min2d(self.rewards.get_batch(batch_idxs))
        next_states_batch = array_min2d(self.next_states.get_batch(batch_idxs))
        terminals_batch = array_min2d(self.terminals.get_batch(batch_idxs))'''
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        # next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)


        return states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch

    def append(self, state, action, reward, next_state, terminal=False, training=True):
        if not training:
            return

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(terminal)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.terminals.clear()

    @property
    def nb_entries(self):
        return len(self.states)



class MemoryV2(object):
    """
    Replay buffer that (optionally) returns:
     - next_actions (if stored)
     - time_steps (if stored)
     - a fixed-length *history* of past actions
     - a fixed-length *history* of next actions
    """
    def __init__(
        self,
        limit,
        observation_shape,
        action_shape,
        next_actions: bool = False,
        time_steps: bool = False,
        history_len: int = 1,
    ):
        self.limit        = limit
        self.history_len  = history_len

        self.states       = RingBuffer(limit, shape=observation_shape)
        self.actions      = RingBuffer(limit, shape=action_shape)
        self.rewards      = RingBuffer(limit, shape=(1,))
        self.next_states  = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.time_steps   = RingBuffer(limit, shape=(1,)) if time_steps else None
        self.terminals    = RingBuffer(limit, shape=(1,))

    def append(
        self,
        state,
        action,
        reward,
        next_state,
        next_action=None,
        terminal=False,
        time_steps=None
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions is not None:
            assert next_action is not None
            self.next_actions.append(next_action)
        self.terminals.append(terminal)
        if self.time_steps is not None:
            assert time_steps is not None
            self.time_steps.append(time_steps)

    @property
    def nb_entries(self):
        return len(self.states)

    def sample(self, batch_size, random_machine=np.random):
        # 1) draw indices
        idxs = random_machine.choice(self.nb_entries, size=batch_size)

        # 2) base batches
        states      = self.states.get_batch(idxs)
        actions     = self.actions.get_batch(idxs)
        rewards     = self.rewards.get_batch(idxs)
        next_states = self.next_states.get_batch(idxs)
        terminals   = self.terminals.get_batch(idxs)

        # 3) optional batches
        next_actions = (
            self.next_actions.get_batch(idxs)
            if self.next_actions is not None else None
        )
        time_steps = (
            self.time_steps.get_batch(idxs)
            if self.time_steps is not None else None
        )

        # 4) build past-action history
        H = self.history_len
        A_shape = self.actions.data.shape[1:]
        past_hist = np.zeros((batch_size, H) + A_shape, dtype=self.actions.data.dtype)
        next_hist = np.zeros_like(past_hist)

        for i, idx in enumerate(idxs):
            # past H actions ending at idx
            for h in range(H):
                hist_idx = idx - (H - 1 - h)
                if hist_idx >= 0:
                    past_hist[i, h] = self.actions[hist_idx]

            # next H actions starting at idx+1
            for h in range(H):
                n_idx = idx + 1 - (H - 1 - h)
                if n_idx >= 0 and n_idx < self.nb_entries:
                    if self.next_actions is not None:
                        next_hist[i, h] = self.next_actions[n_idx]
                    else:
                        next_hist[i, h] = self.actions[n_idx]

        # 5) assemble return tuple
        ret = [states, actions, rewards, next_states]
        if next_actions is not None:
            ret.append(next_actions)
        ret.append(terminals)
        if time_steps is not None:
            ret.append(time_steps)
        # finally histories
        ret.extend([past_hist, next_hist])

        return tuple(ret)


class MemoryNStepReturns(object):
    def __init__(self, limit, observation_shape, action_shape, next_actions=False, time_steps=False, n_step_returns=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.time_steps = RingBuffer(limit, shape=(1,)) if time_steps else None
        self.terminals = RingBuffer(limit, shape=(1,))
        self.n_step_returns = RingBuffer(limit, shape=(1,)) if n_step_returns else None

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        #batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        batch_idxs = random_machine.choice(self.nb_entries, size=batch_size)
        # batch_idxs = random_machine.choice(self.nb_entries, weights=[i/self.nb_entries for i in range(self.nb_entries)], size=batch_size)

        '''states_batch = array_min2d(self.states.get_batch(batch_idxs))
        actions_batch = array_min2d(self.actions.get_batch(batch_idxs))
        rewards_batch = array_min2d(self.rewards.get_batch(batch_idxs))
        next_states_batch = array_min2d(self.next_states.get_batch(batch_idxs))
        terminals_batch = array_min2d(self.terminals.get_batch(batch_idxs))'''
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)
        time_steps = self.time_steps.get_batch(batch_idxs) if self.time_steps is not None else None
        n_step_returns = self.n_step_returns.get_batch(batch_idxs) if self.n_step_returns is not None else None

        ret = [states_batch, actions_batch, rewards_batch, next_states_batch]
        if next_actions is not None:
            ret.append(next_actions)
        ret.append(terminals_batch)
        if time_steps is not None:
            ret.append(time_steps)
        if n_step_returns is not None:
            ret.append(n_step_returns)
        return tuple(ret)

    def append(self, state, action, reward, next_state, next_action=None, terminal=False, time_steps=None,
               n_step_return=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions is not None:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)
        if self.time_steps is not None:
            assert time_steps is not None
            self.time_steps.append(time_steps)
        if self.n_step_returns is not None:
            assert n_step_return is not None
            self.n_step_returns.append(n_step_return)

    @property
    def nb_entries(self):
        return len(self.states)
