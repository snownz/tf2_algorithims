import os, json
import gym
import numpy as np
import tensorflow as tf
from tqdm import trange

from ounoise import ActionNoise, NormalActionNoise, OrnsteinUhlenbeckActionNoise

from gac_net import AutoRegressiveStochasticActor as AIQN
from gac_net import StochasticActor as IQN
from gac_net import Critic, Value
from gac_agent import GACAgent
from ann_utils import categorical_sample

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

class Obj(object):
    def __init__(self):
        pass
class Wrapper(gym.Wrapper):
    def __init__(self, env, normalize_obs):
        super().__init__(env)
        self.env = env
        self.normalize_obs = normalize_obs

    def reset(self):
        state = self.env.reset()
        if self.normalize_obs:
            state = (state - self.observation_space.low)/(self.observation_space.high-self.observation_space.low) * 2.0 - 1.0
        return state

    def step(self, action):
        action = 0.5 * (action + 1) * (self.action_space.high - self.action_space.low) + self.action_space.low
        action = np.clip(action, self.action_space.low, self.action_space.high) # avoid numerical error
        next_state, reward, done, info = self.env.step(action)
        if self.normalize_obs:
            next_state = (next_state - self.observation_space.low)/(self.observation_space.high-self.observation_space.low) * 2.0 - 1.0
        return next_state, reward, done, info

def _reset_noise(agent, a_noise):
    if a_noise is not None:
        a_noise.reset()

def evaluate_policy(policy, env, episodes):
    """
    Run the environment env using policy for episodes number of times.
    Return: average rewards per episode.
    """
    rewards = []
    for _ in range(episodes):
        state = np.float32(env.reset())
        is_terminal = False
        t = 0
        while not is_terminal:
            action = policy.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
            # remove the batch_size dimension if batch_size == 1
            action = tf.squeeze(action, [0]).numpy()
            state, reward, is_terminal, _ = env.step(action)
            state, reward = np.float32(state), np.float32(reward)
            rewards.append(float(reward))
            env.render()
            t+=1
    return rewards

args = Obj()

args.environment = "LunarLanderContinuous-v2"
args.normalize_obs = False
args.noise  = 'normal'
args.gamma = 0.99
args.tau = 5e-3
args.noise_scale = 0.2
args.batch_size = 512
args.epochs = None
args.epoch_cycles = 20
args.rollout_steps = 100
args.T = 10
args.model_path = '/tmp/dpo/'
args.start_timesteps = 10
args.eval_freq = 1000
args.eval_episodes = 5
args.buffer_size = 1000000
args.action_samples = 16
args.visualize = True
args.experiment_name = 'test'
args.print = True
args.actor = 'IQN'
args.normalize_obs = False
args.normalize_rewards = False
args.q_normalization = 0.01
args.mode = 'boltzmann' # ['linear', 'max', 'boltzmann', 'uniform']
args.beta = 1.0
args.num_steps = 2000000

env = Wrapper( gym.make( args.environment ), args.normalize_obs )
eval_env = Wrapper( gym.make( args.environment ), args.normalize_obs )
args.action_dim = env.action_space.shape[0]
args.state_dim = env.observation_space.shape[0]

if args.noise == 'ou':
        noise = OrnsteinUhlenbeckActionNoise( mu = np.zeros( args.action_dim ), sigma = float( args.noise_scale ) * np.ones( args.action_dim ) )
elif args.noise == 'normal':
    noise = NormalActionNoise( mu = np.zeros( args.action_dim ), sigma = float( args.noise_scale ) * np.ones( args.action_dim ) )
else:
    noise = None

base_dir = os.getcwd() + '/models/' + args.environment + '/'
run_number = 0
while os.path.exists(base_dir + str(run_number)):
    run_number += 1
base_dir = base_dir + str(run_number)
os.makedirs(base_dir)

gac = GACAgent(**args.__dict__)
state = env.reset()
results_dict = {
    'train_rewards': [],
    'eval_rewards': [],
    'actor_losses': [],
    'value_losses': [],
    'critic_losses': []
}
episode_steps, episode_rewards = 0, 0 # total steps and rewards for each episode

num_steps = args.num_steps
if num_steps is not None:
    nb_epochs = int(num_steps) // (args.epoch_cycles * args.rollout_steps)
else:
    nb_epochs = 500

_reset_noise( gac, noise )

"""
training loop
"""
average_rewards = [ 0 ] * 100
count = 0
total_steps = 0
train_steps = 0

bar = trange(nb_epochs)
for epoch in bar:
    for cycle in range(args.epoch_cycles):
        for rollout in range(args.rollout_steps):
            """
            Get an action from neural network and run it in the environment
            """
            if total_steps < args.start_timesteps:
                action = tf.expand_dims(env.action_space.sample(), 0)
            else:
                action = gac.select_perturbed_action(
                    tf.convert_to_tensor([state], dtype=tf.float32),
                    noise
                )
            # remove the batch_size dimension if batch_size == 1
            action = tf.squeeze(action, [0]).numpy()
            next_state, reward, is_terminal, _ = env.step(action)
            next_state, reward = np.float32(next_state), np.float32(reward)
            gac.store_transition(state, action, reward, next_state, is_terminal)
            episode_rewards += reward

            # check if game is terminated to decide how to update state, episode_steps,
            # episode_rewards
            #env.render()
            if is_terminal:
                state = np.float32(env.reset())
                results_dict['train_rewards'].append(
                    (total_steps, episode_rewards)
                )
                episode_steps = 0
                episode_rewards = 0
                _reset_noise(gac, noise)
            else:
                state = next_state
                episode_steps += 1

            # evaluate
            if total_steps % args.eval_freq == 0:
                eval_rewards = evaluate_policy(gac, eval_env, args.eval_episodes)
                eval_reward = sum(eval_rewards) / args.eval_episodes
                eval_variance = float(np.var(eval_rewards))
                results_dict['eval_rewards'].append({
                    'total_steps': total_steps,
                    'train_steps': train_steps,
                    'average_eval_reward': eval_reward,
                    'eval_reward_variance': eval_variance
                })
                with open('results.txt', 'w') as file:
                    file.write(json.dumps(results_dict['eval_rewards']))

            total_steps += 1

            average_rewards.pop( 0 )
            average_rewards.append( reward )

            bar.set_description('average_rewards: {} - Steps: {} - TSteps: {}'.format( np.mean( average_rewards ), total_steps, train_steps ) )
            bar.refresh() # to show immediately the update

        # train
        if gac.replay.size >= args.batch_size:
            for _ in range(args.T):
                gac.train_one_step()
                train_steps += 1

with open('results.txt', 'w') as file:
    file.write(json.dumps(results_dict))

#utils.save_model(gac.actor, base_dir)