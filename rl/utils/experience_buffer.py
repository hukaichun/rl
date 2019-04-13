import numpy as np


def range_mod(start, num, mod):
        return [(start+i)%mod for i in range(num)]


class trajecories:
    def __init__(self,
        state_shape, 
        action_shape,
        max_length,
        capacity,
        parall_num = 1,
        state_dtype = np.float64
    ):
        self._ith = 0
        self._regular_buff_range = capacity
        self._swap_range         = parall_num
        self._max_length         = max_length

        total_capacity = capacity+parall_num

        state_buff_shape  = (total_capacity, max_length+1) + state_shape
        action_buff_shape = (total_capacity, max_length) + action_shape
        info_buff_shape   = (total_capacity, max_length)

        self._tail_idx  = np.zeros(capacity+1,      dtype=np.int32)
        self._states    = np.zeros(state_buff_shape,  dtype=state_dtype)
        self._actions   = np.zeros(action_buff_shape, dtype=np.float32)
        self._probs     = np.zeros(info_buff_shape,   dtype=np.float32)
        self._flag      = np.zeros(info_buff_shape,   dtype=np.bool) 

        self._full = False

    def append(self, state, action, prob, next_state, done_flag):
        capacity = self._regular_buff_range
        tail_idx = self._tail_idx[capacity]

        if tail_idx == 0:
            self._states[capacity:, tail_idx] = state
        else:
            bool_mask = self._flag[capacity:, tail_idx-1]
            if np.any(bool_mask):
                tmpSlice = self._states[capacity:, tail_idx]
                tmpSlice[bool_mask] = state[bool_mask]

        self._states[capacity:, tail_idx+1] = next_state
        self._actions[capacity:, tail_idx]  = action
        self._probs[capacity:, tail_idx]    = prob
        self._flag[capacity:, tail_idx]     = done_flag

        self._tail_idx[capacity]+=1
        if self._tail_idx[capacity] == self._max_length:
            self._tail_idx[capacity] = 0
            idxs = range_mod(self._ith, self._swap_range, capacity)
            self._ith = idxs[-1]+1
            self._states[idxs]  = self._states[capacity:]
            self._actions[idxs] = self._actions[capacity:]
            self._probs[idxs]   = self._probs[capacity:]
            self._flag[idxs]    = self._flag[capacity:]
            if capacity in idxs:
                self._full = True 

            


if __name__ == "__main__":

    state_shape = (2,)
    action_shape = (2,)
    max_length = 5
    capacity = 10
    parall_num = 2




    qqq = trajecories(state_shape, action_shape, max_length, capacity, parall_num)

    print(qqq._states,'\n')
    for i in range(11):
        state = np.zeros((parall_num,)+state_shape)
        state[:,0] = i
        state[:,1] = -1

        for j in range(5):
            
            action = np.asarray((i,j))
            prob = 1./(0.5+i+j)*np.ones((parall_num, 1))
            term = (j==3)* np.ones((parall_num, 1),dtype=np.bool)
            next_state = np.zeros((parall_num,)+state_shape)
            next_state[:,0] = i
            next_state[:,1] = j-(j==3)

            qqq.append(state, action, prob, next_state, term)
            state = next_state
    print(qqq._states)



