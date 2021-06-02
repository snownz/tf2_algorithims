import os, json
import gym
import numpy as np
from tqdm import trange
from multi_env_warper import MultiEnv

from dqn_agent import NQRDqnMultiAgent

def np_onehot(v, s):
    a = np.zeros( s )
    a[v] = 1
    return a

map0 = [ 0, 1, 2, 3 ]
map1 = [ 3, 2, 1, 0 ]
map2 = [ 1, 2, 3, 0 ]
map3 = [ 2, 3, 0, 1 ]
map4 = [ 2, 3, 1, 0 ]
map5 = [ 3, 1, 2, 0 ]

env_params = [
    # {  'id': '1', 'name': 'gym_rocketlander:rocketlander-v0', 'recurrent': True,  'rw_func': lambda x, step, done: x * 10.0, 'st_func': lambda x: x },
    {  'id': '0', 'name': 'LunarLander-v2', 'recurrent': False, 'rw_func': lambda x, step, done: x / 10.0, 'st_func': lambda x: x, 'ac_func': lambda x: map0[x] },
    {  'id': '1', 'name': 'LunarLander-v2', 'recurrent': False, 'rw_func': lambda x, step, done: x / 10.0, 'st_func': lambda x: x, 'ac_func': lambda x: map1[x] },
    {  'id': '2', 'name': 'LunarLander-v2', 'recurrent': False, 'rw_func': lambda x, step, done: x / 10.0, 'st_func': lambda x: x, 'ac_func': lambda x: map2[x] },
    {  'id': '3', 'name': 'LunarLander-v2', 'recurrent': True, 'rw_func': lambda x, step, done: x / 10.0, 'st_func': lambda x: x, 'ac_func': lambda x: map3[x] },
    {  'id': '4', 'name': 'LunarLander-v2', 'recurrent': True, 'rw_func': lambda x, step, done: x / 10.0, 'st_func': lambda x: x, 'ac_func': lambda x: map4[x] },
    {  'id': '5', 'name': 'LunarLander-v2', 'recurrent': True, 'rw_func': lambda x, step, done: x / 10.0, 'st_func': lambda x: x, 'ac_func': lambda x: map5[x] },
]

exec_params = {

    # training params
    'experiment_name': 'test_multi_env_v2',
    'epochs': 10000,
    'batch_size': 2048,
    'rollout_steps': 1000,
    'buffer_size': 50000,
    'sequence': 16,
    'train_iter': 40,

    # global memory params
    'gm': 2,
    'gn': 2,

    # general params
    'train': True,
    'load': False,
}

env = MultiEnv( env_params, exec_params['sequence'], exec_params['gm'], exec_params['gn'] )

dqn = NQRDqnMultiAgent( env.envs, exec_params['buffer_size'], exec_params['batch_size'], exec_params['experiment_name'], 
                        exec_params['sequence'], 32, 64, 8 )

base_dir = os.getcwd() + '/models/' + exec_params['experiment_name'] + '/'
if not os.path.exists( base_dir ): os.makedirs( base_dir )

if exec_params['load']: dqn.restore_training( base_dir + 'training/' )

env.reset()

total_steps = 0
train_steps = dqn.t_step
bar = trange( exec_params['epochs'] )
for epoch in bar:
    
    for rollout in range( exec_params['rollout_steps'] ):

        actions = dqn.act( env.envs, total_steps, exec_params['train'] )
        
        env.step( actions )
        dqn.step( env.envs )
        env.update()

        total_steps += 1

        bar.set_description('Steps: {} - TSteps: {}'.format( total_steps, train_steps ) )
        bar.refresh()

        if not exec_params['train']:
            env.render()

    if epoch >= 2 and exec_params['train']:
        for _ in range(exec_params['train_iter']):
            dqn.learn()
            train_steps += 1
        env.reset()
    
    if exec_params['train']:
        dqn.save_training( base_dir + 'training/' )