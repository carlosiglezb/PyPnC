from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
import math
from copy import copy
import json
import multiprocessing as mp
from tqdm import tqdm


def pretty_print(ob):
    print(json.dumps(ob, indent=4))


def euler_to_rot(angles):
    # Euler ZYX to Rot
    # Note that towr has (x, y, z) order
    x = angles[0]
    y = angles[1]
    z = angles[2]
    ret = np.array([
        np.cos(y) * np.cos(z),
        np.cos(z) * np.sin(x) * np.sin(y) - np.cos(x) * np.sin(z),
        np.sin(x) * np.sin(z) + np.cos(x) * np.cos(z) * np.sin(y),
        np.cos(y) * np.sin(z),
        np.cos(x) * np.cos(z) + np.sin(x) * np.sin(y) * np.sin(z),
        np.cos(x) * np.sin(y) * np.sin(z) - np.cos(z) * np.sin(x), -np.sin(y),
        np.cos(y) * np.sin(x),
        np.cos(x) * np.cos(y)
    ]).reshape(3, 3)
    return np.copy(ret)


def vec_to_roll_pitch(vec):
    # return zeros if vector is a point
    if np.linalg.norm(vec) < 1e-3:
        return 0.0, 0.0

    vec_copy = np.copy(vec)
    vec_copy = vec_copy / np.linalg.norm(vec)
    roll = np.arcsin(-vec_copy[1])
    pitch = np.arctan2(vec_copy[0], vec_copy[2])
    return roll, pitch

def quat_to_rot(quat):
    """
    Parameters
    ----------
    quat (np.array): scalar last quaternion

    Returns
    -------
    ret (np.array): SO3

    """
    return np.copy((R.from_quat(quat)).as_matrix())


def rot_to_quat(rot):
    """
    Parameters
    ----------
    rot (np.array): SO3

    Returns
    -------
    quat (np.array): scalar last quaternion

    """
    return np.copy(R.from_matrix(rot).as_quat())


def quat_to_exp(quat):
    img_vec = np.array([quat[0], quat[1], quat[2]])
    w = quat[3]
    theta = 2.0 * np.arcsin(
        np.sqrt(img_vec[0] * img_vec[0] + img_vec[1] * img_vec[1] +
                img_vec[2] * img_vec[2]))

    if np.abs(theta) < 1e-4:
        return np.zeros(3)
    ret = img_vec / np.sin(theta / 2.0)

    return np.copy(ret * theta)


def exp_to_quat(exp):
    theta = np.sqrt(exp[0] * exp[0] + exp[1] * exp[1] + exp[2] * exp[2])
    ret = np.zeros(4)
    if theta > 1e-4:
        ret[0] = np.sin(theta / 2.0) * exp[0] / theta
        ret[1] = np.sin(theta / 2.0) * exp[1] / theta
        ret[2] = np.sin(theta / 2.0) * exp[2] / theta
        ret[3] = np.cos(theta / 2.0)
    else:
        ret[0] = 0.5 * exp[0]
        ret[1] = 0.5 * exp[1]
        ret[2] = 0.5 * exp[2]
        ret[3] = 1.0
    return np.copy(ret)


def weighted_pinv(A, W, rcond=1e-15):
    return np.dot(
        W,
        np.dot(A.transpose(),
               np.linalg.pinv(np.dot(np.dot(A, W), A.transpose()), rcond)))


def get_sinusoid_trajectory(start_time, mid_point, amp, freq, eval_time):
    dim = amp.shape[0]
    p, v, a = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    p = amp * np.sin(2 * np.pi * freq * (eval_time - start_time)) + mid_point
    v = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq *
                                        (eval_time - start_time))
    a = -amp * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq *
                                              (eval_time - start_time))

    return p, v, a


def normalize_data(data):
    mean = np.mean(np.stack(data, axis=0), axis=0)
    std = np.std(np.stack(data, axis=0), axis=0)

    return mean, std, normalize(data, mean, std)


def normalize(x, mean, std):
    assert std.shape == mean.shape
    if type(x) is list:
        assert x[0].shape == mean.shape
        ret = []
        for val in x:
            ret.append((val - mean) / std)
        return ret
    else:
        assert x.shape == mean.shape
        return (x - mean) / std


def denormalize(x, mean, std):
    assert std.shape == mean.shape
    if type(x) is list:
        assert x[0].shape == mean.shape
        ret = []
        for val in x:
            ret.append(val * std + mean)
        return ret
    else:
        assert x.shape == mean.shape
        return x * std + mean


def print_attrs(ob):
    attr = vars(ob)
    print(", \n".join("%s: %s" % item for item in attr.items()))


def try_multiprocess(args_list, num_cpu, f, max_timeouts=1):
    """
    Multiprocessing wrapper function.
    """
    if max_timeouts == 0:
        return None

    if num_cpu == 1:
        return [f(args_list)]
    else:
        pool = mp.Pool(processes=num_cpu,
                       maxtasksperchild=1,
                       initargs=(mp.RLock(), ),
                       initializer=tqdm.set_lock)
        pruns = []
        for i in range(num_cpu):
            rseed = np.random.randint(1000000)
            pruns.append(pool.apply_async(f, args=(args_list + [rseed, i], )))
        try:
            results = [p.get(timeout=36000) for p in pruns]
        except Exception as e:
            print(str(e))
            print('WARNING: error raised in multiprocess, trying again')

            pool.close()
            pool.terminate()
            pool.join()

            return try_multiprocess(args_list, num_cpu, f, max_timeouts - 1)

        pool.close()
        pool.terminate()
        pool.join()

    return results


def prevent_quat_jump(quat_des, quat_act):
    # print("quat_des:",quat_des)
    # print("quat_act:",quat_act)
    a = quat_des - quat_act
    b = quat_des + quat_act
    if np.linalg.norm(a) > np.linalg.norm(b):
        new_quat_act = -quat_act
    else:
        new_quat_act = quat_act

    return new_quat_act


def is_colliding_3d(start, goal, min, max, threshold, N):
    for i in range(3):
        for j in range(N):
            p = start[i] + (goal[i] - start[i]) * j / N
            if min[i] + np.abs(threshold[i]) <= p and p <= max[i] - np.abs(
                    threshold[i]):
                return True
    return False


class GridLocation(object):
    def __init__(self, delta):
        """
        Parameters
        ----------
        delta (np.array): 1d array
        """
        self._dim = delta.shape[0]
        self._delta = np.copy(delta)

    def get_grid_idx(self, pos):
        """
        Parameters
        ----------
        pos (np.array): 1d array

        Returns
        -------
        v (double or tuple): idx
        """
        v = np.zeros(self._dim, dtype=int)
        for i in range(self._dim):
            v[i] = pos[i] // self._delta[i]

        if self._dim == 1:
            return v[0]
        else:
            return tuple(v)

    def get_boundaries(self, idx):
        """
        Parameters
        ----------
        idx (np.array): 1d array of integer

        Returns
        -------
        v (np.array): 1d array of boundaries [min, max, min, max]
        """

        bds = np.zeros(self._dim * 2, dtype=float)
        for i in range(self._dim):
            bds[2 * i] = idx[i] * self._delta[i]
            bds[2 * i + 1] = (idx[i] + 1) * self._delta[i]

        return bds

    def get_center(self, idx):
        """
        Parameters
        ----------
        idx (np.array): 1d array of integer

        Returns
        -------
        c (np.array): center
        """

        if self._dim == 1:
            bds = self.get_boundaries(idx)
            return (bds[0] + bds[1]) / 2.
        else:
            bds = self.get_boundaries(idx)
            return np.array([(bds[0] + bds[1]) / 2., (bds[2] + bds[3]) / 2.])


def trajectory_scaler(prev_trajectory, N_scaled_horizon, time_indices, current_time, control_dt):

    # model parameters
    nx = len(prev_trajectory[0])

    # initialize by assuming the first values for all N_scaled_horizon values
    scaled_trajectory = np.repeat(np.reshape(prev_trajectory[0], (1, nx)),
                                  N_scaled_horizon, axis=0)

    # find t' & index i'
    ti_idx = 1
    for idx in range(1, N_scaled_horizon):
        curr_traj_time = current_time + idx * control_dt
        if curr_traj_time > time_indices[ti_idx]:
            ti_idx += 1
        time_since_prev_ti = curr_traj_time - time_indices[ti_idx-1]

        # Note: if the trajectories can be matched one-to-one by skipping certain nodes, e.g.,
        # idx_prev_traj == integer, this can be simplified, e.g.:
        # idx_prev_traj = int(idx_prev_traj)
        # scaled_trajectory[nx * idx:nx * (idx + 1)] \
        #     = prev_trajectory[nx * idx_prev_traj:nx * (idx_prev_traj + 1)]

        # assuming idx_prev_traj ~= integer:
        x_l = prev_trajectory[ti_idx-1]
        x_u = prev_trajectory[ti_idx]

        # corrected values
        dx = x_u - x_l

        # linear interpolation
        mpc_dt = time_indices[ti_idx] - time_indices[ti_idx-1]
        decimal = np.around(time_since_prev_ti / mpc_dt, decimals=5)
        x_interpol = x_l + decimal*dx

        # fill up the initial guess
        scaled_trajectory[idx] = x_interpol

    return scaled_trajectory
