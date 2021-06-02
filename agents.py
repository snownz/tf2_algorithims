import tensorflow as tf
from custom_rl import SimplePPONetwork, NaluPPONetwork, NaluAdavancedPPONetwork
import copy
import numpy as np
from utils import save_checkpoint, restore_checkpoint
from multi_env_warper import Environment
from multiprocessing import Pipe
import os

class SimplePPOAgent():

    def __init__(self, env):

        self.env = env
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 10000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 100 # when average score is above 0 model will be saved
        self.max_steps = 200 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 # training epochs
        self.shuffle = False
        self.Training_batch = 1000

        self.replay_count = 0
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        self.model = SimplePPONetwork( self.action_size, 'ppo', self.lr )

        self.model( tf.zeros( [ 1, self.state_size[0] ] ) )
    
    def act(self, state):
        
        prediction = self.model.probs( state ).numpy()[0]
        action = np.random.choice( self.action_size, p = prediction )
        action_onehot = np.zeros( [ self.action_size ] )
        action_onehot[ action ] = 1
        return action, action_onehot, prediction
    
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        
        deltas = [ r + gamma * (1 - d) * nv - v for r, d, nv, v in zip( rewards, dones, next_values, values ) ]
        deltas = np.stack( deltas )
        gaes = copy.deepcopy( deltas )
        
        for t in reversed( range( len( deltas ) - 1 ) ):
            gaes[t] = gaes[t] + ( 1 - dones[t] ) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        
        return np.vstack(gaes), np.vstack(target)
    
    def replay(self, states, actions, rewards, predictions, dones, next_states):
        
        states = np.vstack( states )
        next_states = np.vstack( next_states )
        actions = np.vstack( actions )
        predictions = np.vstack( predictions )

        # Get Critic network predictions 
        values = self.model.value( states )
        next_values = self.model.value( next_states )

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes( rewards, dones, np.squeeze( values ), np.squeeze( next_values ) )
        
        # training Actor and Critic networks
        self.model.train( states, values, target, advantages, predictions, actions, rewards, epochs = self.epochs, shuffle=self.shuffle )

        with self.model.train_summary_writer.as_default():
            
            tf.summary.scalar( 'Data/actor_loss_per_replay', self.model.a_loss.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/critic_loss_per_replay', self.model.c_loss.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/reward_per_replay', self.model.reward.result(), step = self.replay_count )

        self.replay_count += 1

    def run(self): # train only when episode is finished
        
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score = False, 0
        
        while True:
            
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            t_steps = 0
            while not done:

                t_steps += 1
                # self.env.render()
                # Actor picks an action
                action, action_onehot, prediction = self.act( state )
                
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step( action )
                
                # Memorize (state, action, reward) for training
                states.append( state )
                next_states.append( np.reshape( next_state, [ 1, self.state_size[0]] ) )
                actions.append( action_onehot )
                rewards.append( reward )
                dones.append( done )
                predictions.append( prediction )
                
                # Update current state
                state = np.reshape( next_state, [ 1, self.state_size[0] ] )
                score += reward
                
                if done or t_steps >= self.max_steps:
                    self.episode += 1

                    avg = self.model.reward.result()
                    if avg >= self.max_average: self.lr *= 0.95

                    print("episode: {}/{}, score: {}".format(self.episode, self.EPISODES, score), end="\r", flush=True)
                    
                    with self.model.train_summary_writer.as_default():

                        tf.summary.scalar( f'Workers:{1}/score_per_episode', score, step = self.episode )
                        tf.summary.scalar( f'Workers:{1}/learning_rate', self.lr, step = self.episode )
                    
                    self.replay( states, actions, rewards, predictions, dones, next_states )

                    state, done, score = self.env.reset(), False, 0
                    state = np.reshape( state, [ 1, self.state_size[0] ] )

                    if self.episode%100 == 0:
                        self.save_training( os.getcwd() + '/models/ppo/' )

                    break

            if self.episode >= self.EPISODES:
                break
        self.env.close()

    def run_batch(self): # train every self.Training_batch episodes
        
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score = False, 0
        
        while True:
            
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            for t in range(self.Training_batch):
                
                # self.env.render()

                avg = self.model.reward.result()
                if avg >= self.max_average: self.lr *= 0.95
                
                # Actor picks an action
                action, action_onehot, prediction = self.act(state)
                
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action)
                
                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                
                # Update current state
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                
                if done:
                    self.episode += 1
                    print("episode: {}/{}, score: {}".format(self.episode, self.EPISODES, score ), end="\r", flush=True)
                    
                    with self.model.train_summary_writer.as_default():

                        tf.summary.scalar( f'Workers:{1}/score_per_episode', score, step = self.episode )
                        tf.summary.scalar( f'Workers:{1}/learning_rate', self.lr, step = self.episode )

                    state, done, score = self.env.reset(), False, 0
                    state = np.reshape(state, [1, self.state_size[0]])
                    
            self.replay(states, actions, rewards, predictions, dones, next_states)
            self.save_training( os.getcwd() + '/models/ppo/' )
            
            if self.episode >= self.EPISODES:
                break
        
        self.env.close()  
    
    def run_multiprocesses(self, num_worker = 4):
        
        works, parent_conns, child_conns = [], [], []
        
        for idx in range( num_worker ):
            parent_conn, child_conn = Pipe()
            work = Environment( idx, child_conn, 'LunarLander-v2', self.state_size[0], self.action_size, False )
            work.start()
            works.append( work )
            parent_conns.append( parent_conn )
            child_conns.append( child_conn )

        states =        [ [] for _ in range( num_worker ) ]
        next_states =   [ [] for _ in range( num_worker ) ]
        actions =       [ [] for _ in range( num_worker ) ]
        rewards =       [ [] for _ in range( num_worker ) ]
        dones =         [ [] for _ in range( num_worker ) ]
        predictions =   [ [] for _ in range( num_worker ) ]
        score =         [  0 for _ in range( num_worker ) ]

        state = [ 0 for _ in range( num_worker ) ]
        for worker_id, parent_conn in enumerate( parent_conns ):
            state[ worker_id ] = parent_conn.recv()

        while self.episode < self.EPISODES:
            
            predictions_list = self.model.probs( np.reshape( state, [ num_worker, self.state_size[0] ] ) ).numpy()
            actions_list = [ np.random.choice( self.action_size, p = i ) for i in predictions_list ]

            for worker_id, parent_conn in enumerate( parent_conns ):
                
                parent_conn.send( actions_list[ worker_id ] )
                action_onehot = np.zeros( [ self.action_size ] )
                action_onehot[ actions_list[ worker_id ] ] = 1
                actions[ worker_id ].append( action_onehot )
                predictions[ worker_id ].append( predictions_list[ worker_id ] )

            for worker_id, parent_conn in enumerate( parent_conns ):
                
                next_state, reward, done, _ = parent_conn.recv()

                states[ worker_id ].append( state[ worker_id ] )
                next_states[ worker_id ].append( next_state )
                rewards[ worker_id ].append( reward )
                dones[ worker_id ].append( done )
                state[ worker_id ] = next_state
                score[ worker_id ] += reward

                if done:

                    print("episode: {}/{}, worker: {}, score: {}".format(self.episode, self.EPISODES, worker_id, score[worker_id]), end="\r", flush=True)

                    with self.model.train_summary_writer.as_default():

                        tf.summary.scalar( 'Workers:{}/score_per_episode'.format(worker_id), score[worker_id], step = self.episode )
                        tf.summary.scalar( 'Workers:{}/learning_rate'.format(worker_id), self.lr, step = self.episode )

                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1
                        
            for worker_id in range(num_worker):
                
                if len(states[worker_id]) >= self.Training_batch:
                    
                    self.replay( states[ worker_id ], actions[ worker_id ], rewards[ worker_id ], predictions[ worker_id ], dones[ worker_id ], next_states[ worker_id ] )
                    self.save_training( os.getcwd() + '/models/ppo/' )

                    states[worker_id]      = []
                    next_states[worker_id] = []
                    actions[worker_id]     = []
                    rewards[worker_id]     = []
                    dones[worker_id]       = []
                    predictions[worker_id] = []

        # terminating processes after while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()

    def test(self, test_episodes = 100):
        self.restore_training( os.getcwd() + '/models/ppo/' )
        for e in range(100):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax( self.model.probs(state).numpy()[0] )
                state, reward, done, _ = self.env.step( action )
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, test_episodes, score))
                    break
        self.env.close()

    def save_training(self, directory):
        save_checkpoint( self.model, directory + 'local', self.replay_count )

    def restore_training(self, directory):
        self.replay_count = restore_checkpoint( self.model, directory + 'local' )


class NaluPPOAgent():

    def __init__(self, env):

        self.env = env
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 10000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.max_steps = 200 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 # training epochs
        self.shuffle = False
        self.Training_batch = 256

        self.replay_count = 0
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        self.model = NaluPPONetwork( self.action_size, 'nalu_ppo', self.lr )

        self.model( tf.zeros( [ 1, self.state_size[0] ] ) )
    
    def act(self, state):
        
        prediction = self.model.probs( state ).numpy()[0]
        action = np.random.choice( self.action_size, p = prediction )
        action_onehot = np.zeros( [ self.action_size ] )
        action_onehot[ action ] = 1
        return action, action_onehot, prediction
    
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        
        deltas = [ r + gamma * (1 - d) * nv - v for r, d, nv, v in zip( rewards, dones, next_values, values ) ]
        deltas = np.stack( deltas )
        gaes = copy.deepcopy( deltas )
        
        for t in reversed( range( len( deltas ) - 1 ) ):
            gaes[t] = gaes[t] + ( 1 - dones[t] ) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        
        return np.vstack(gaes), np.vstack(target)
    
    def replay(self, states, actions, rewards, predictions, dones, next_states):
        
        states = np.vstack( states )
        next_states = np.vstack( next_states )
        actions = np.vstack( actions )
        predictions = np.vstack( predictions )

        # Get Critic network predictions 
        values = self.model.value( states )
        next_values = self.model.value( next_states )

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes( rewards, dones, np.squeeze( values ), np.squeeze( next_values ) )
        
        # training Actor and Critic networks
        self.model.train( states, values, target, advantages, predictions, actions, rewards, epochs = self.epochs, shuffle=self.shuffle )

        with self.model.train_summary_writer.as_default():
            
            tf.summary.scalar( 'Data/actor_loss_per_replay', self.model.a_loss.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/critic_loss_per_replay', self.model.c_loss.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/reward_per_replay', self.model.reward.result(), step = self.replay_count )

        self.replay_count += 1

    def run(self): # train only when episode is finished
        
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score = False, 0
        
        while True:
            
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            t_steps = 0
            while not done:

                t_steps += 1
                # self.env.render()
                # Actor picks an action
                action, action_onehot, prediction = self.act( state )
                
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step( action )
                
                # Memorize (state, action, reward) for training
                states.append( state )
                next_states.append( np.reshape( next_state, [ 1, self.state_size[0]] ) )
                actions.append( action_onehot )
                rewards.append( reward )
                dones.append( done )
                predictions.append( prediction )
                
                # Update current state
                state = np.reshape( next_state, [ 1, self.state_size[0] ] )
                score += reward
                
                if done or t_steps >= self.max_steps:
                    self.episode += 1

                    avg = self.model.reward.result()
                    if avg >= self.max_average: self.lr *= 0.95

                    print("episode: {}/{}, score: {}".format(self.episode, self.EPISODES, score), end="\r", flush=True)
                    
                    with self.model.train_summary_writer.as_default():

                        tf.summary.scalar( f'Workers:{1}/score_per_episode', score, step = self.episode )
                        tf.summary.scalar( f'Workers:{1}/learning_rate', self.lr, step = self.episode )
                    
                    self.replay( states, actions, rewards, predictions, dones, next_states )

                    state, done, score = self.env.reset(), False, 0
                    state = np.reshape( state, [ 1, self.state_size[0] ] )

                    if self.episode%100 == 0:
                        self.save_training( os.getcwd() + '/models/ppo/' )

                    break

            if self.episode >= self.EPISODES:
                break
        self.env.close()

    def run_batch(self): # train every self.Training_batch episodes
        
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score = False, 0
        
        while True:
            
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            for t in range(self.Training_batch):
                
                # self.env.render()

                avg = self.model.reward.result()
                if avg >= self.max_average: self.lr *= 0.95
                
                # Actor picks an action
                action, action_onehot, prediction = self.act(state)
                
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action)
                
                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                
                # Update current state
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                
                if done:
                    self.episode += 1
                    print("episode: {}/{}, score: {}".format(self.episode, self.EPISODES, score ), end="\r", flush=True)
                    
                    with self.model.train_summary_writer.as_default():

                        tf.summary.scalar( f'Workers:{1}/score_per_episode', score, step = self.episode )
                        tf.summary.scalar( f'Workers:{1}/learning_rate', self.lr, step = self.episode )

                    state, done, score = self.env.reset(), False, 0
                    state = np.reshape(state, [1, self.state_size[0]])
                    
            self.replay(states, actions, rewards, predictions, dones, next_states)
            self.save_training( os.getcwd() + '/models/ppo/' )
            
            if self.episode >= self.EPISODES:
                break
        
        self.env.close()  
    
    def run_multiprocesses(self, num_worker = 4):
        
        works, parent_conns, child_conns = [], [], []
        
        for idx in range( num_worker ):
            parent_conn, child_conn = Pipe()
            work = Environment( idx, child_conn, 'LunarLander-v2', self.state_size[0], self.action_size, False )
            work.start()
            works.append( work )
            parent_conns.append( parent_conn )
            child_conns.append( child_conn )

        states =        [ [] for _ in range( num_worker ) ]
        next_states =   [ [] for _ in range( num_worker ) ]
        actions =       [ [] for _ in range( num_worker ) ]
        rewards =       [ [] for _ in range( num_worker ) ]
        dones =         [ [] for _ in range( num_worker ) ]
        predictions =   [ [] for _ in range( num_worker ) ]
        score =         [  0 for _ in range( num_worker ) ]

        state = [ 0 for _ in range( num_worker ) ]
        for worker_id, parent_conn in enumerate( parent_conns ):
            state[ worker_id ] = parent_conn.recv()

        while self.episode < self.EPISODES:
            
            predictions_list = self.model.probs( np.reshape( state, [ num_worker, self.state_size[0] ] ) ).numpy()
            actions_list = [ np.random.choice( self.action_size, p = i ) for i in predictions_list ]

            for worker_id, parent_conn in enumerate( parent_conns ):
                
                parent_conn.send( actions_list[ worker_id ] )
                action_onehot = np.zeros( [ self.action_size ] )
                action_onehot[ actions_list[ worker_id ] ] = 1
                actions[ worker_id ].append( action_onehot )
                predictions[ worker_id ].append( predictions_list[ worker_id ] )

            for worker_id, parent_conn in enumerate( parent_conns ):
                
                next_state, reward, done, _ = parent_conn.recv()

                states[ worker_id ].append( state[ worker_id ] )
                next_states[ worker_id ].append( next_state )
                rewards[ worker_id ].append( reward )
                dones[ worker_id ].append( done )
                state[ worker_id ] = next_state
                score[ worker_id ] += reward

                if done:

                    print("episode: {}/{}, worker: {}, score: {}".format(self.episode, self.EPISODES, worker_id, score[worker_id]), end="\r", flush=True)

                    with self.model.train_summary_writer.as_default():

                        tf.summary.scalar( 'Workers:{}/score_per_episode'.format(worker_id), score[worker_id], step = self.episode )
                        tf.summary.scalar( 'Workers:{}/learning_rate'.format(worker_id), self.lr, step = self.episode )

                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1
                        
            # for worker_id in range(num_worker):
                
            #     if len(states[worker_id]) >= self.Training_batch:
                    
            #         self.replay( states[ worker_id ], actions[ worker_id ], rewards[ worker_id ], predictions[ worker_id ], dones[ worker_id ], next_states[ worker_id ] )
            #         self.save_training( os.getcwd() + '/models/naluppo2/' )

            #         states[worker_id]      = []
            #         next_states[worker_id] = []
            #         actions[worker_id]     = []
            #         rewards[worker_id]     = []
            #         dones[worker_id]       = []
            #         predictions[worker_id] = []

            l_states =       []
            l_next_states =  []
            l_actions =      []
            l_rewards =      []
            l_dones =        []
            l_predictions =  []
            for worker_id in range(num_worker):
                
                if len(states[worker_id]) >= self.Training_batch:

                    l_states.extend( states[worker_id] )
                    l_next_states.extend( next_states[worker_id] )
                    l_actions.extend( actions[worker_id] )
                    l_rewards.extend( rewards[worker_id] )
                    l_dones.extend( dones[worker_id] )
                    l_predictions.extend( predictions[worker_id] )
                    
                    states[worker_id]      = []
                    next_states[worker_id] = []
                    actions[worker_id]     = []
                    rewards[worker_id]     = []
                    dones[worker_id]       = []
                    predictions[worker_id] = []

            if len( l_states ) > 0:
                self.replay( l_states, l_actions, l_rewards, l_predictions, l_dones, l_next_states )

        # terminating processes after while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()

    def test(self, test_episodes = 100):
        self.restore_training( os.getcwd() + '/models/naluppo2/' )
        for e in range(100):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax( self.model.probs(state).numpy()[0] )
                state, reward, done, _ = self.env.step( action )
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, test_episodes, score))
                    break
        self.env.close()

    def save_training(self, directory):
        save_checkpoint( self.model, directory + 'local', self.replay_count )

    def restore_training(self, directory):
        self.replay_count = restore_checkpoint( self.model, directory + 'local' )


class NaluAdvancedPPOAgent():

    def __init__(self, env):

        self.env = env
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 10000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.max_steps = 200 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 # training epochs
        self.shuffle = False
        self.Training_batch = 128

        self.replay_count = 0
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        self.model = NaluAdavancedPPONetwork( self.action_size, 'nalu_ppo_adv', self.lr )

        self.model( tf.zeros( [ 1, self.state_size[0] ] ) )
    
    def act(self, state):
        
        prediction = self.model.probs( state ).numpy()[0]
        action = np.random.choice( self.action_size, p = prediction )
        action_onehot = np.zeros( [ self.action_size ] )
        action_onehot[ action ] = 1
        return action, action_onehot, prediction
    
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        
        deltas = [ r + gamma * (1 - d) * nv - v for r, d, nv, v in zip( rewards, dones, next_values, values ) ]
        deltas = np.stack( deltas )
        gaes = copy.deepcopy( deltas )
        
        for t in reversed( range( len( deltas ) - 1 ) ):
            gaes[t] = gaes[t] + ( 1 - dones[t] ) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        
        return np.vstack(gaes), np.vstack(target)
    
    def replay(self, states, actions, rewards, predictions, dones, next_states):
        
        states = np.vstack( states )
        next_states = np.vstack( next_states )
        actions = np.vstack( actions )
        predictions = np.vstack( predictions )

        # Get Critic network predictions 
        values = self.model.value( states )
        next_values = self.model.value( next_states )

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes( rewards, dones, np.squeeze( values ), np.squeeze( next_values ) )
        
        # training Actor and Critic networks
        self.model.train( states, values, target, advantages, predictions, actions, rewards, epochs = self.epochs, shuffle=self.shuffle )

        with self.model.train_summary_writer.as_default():
            
            tf.summary.scalar( 'Data/actor_loss_per_replay', self.model.a_loss.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/critic_loss_per_replay', self.model.c_loss.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/reward_per_replay', self.model.reward.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/learning_rate_per_replay', self.model.lr( self.replay_count * self.epochs ).numpy(), step = self.episode )


        self.replay_count += 1

    def run(self): # train only when episode is finished
        
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score = False, 0
        
        while True:
            
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            t_steps = 0
            while not done:

                t_steps += 1
                # self.env.render()
                # Actor picks an action
                action, action_onehot, prediction = self.act( state )
                
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step( action )
                
                # Memorize (state, action, reward) for training
                states.append( state )
                next_states.append( np.reshape( next_state, [ 1, self.state_size[0]] ) )
                actions.append( action_onehot )
                rewards.append( reward )
                dones.append( done )
                predictions.append( prediction )
                
                # Update current state
                state = np.reshape( next_state, [ 1, self.state_size[0] ] )
                score += reward
                
                if done or t_steps >= self.max_steps:
                    self.episode += 1

                    avg = self.model.reward.result()
                    if avg >= self.max_average: self.lr *= 0.95

                    print("episode: {}/{}, score: {}".format(self.episode, self.EPISODES, score), end="\r", flush=True)
                    
                    with self.model.train_summary_writer.as_default():

                        tf.summary.scalar( f'Workers:{1}/score_per_episode', score, step = self.episode )
                        tf.summary.scalar( f'Workers:{1}/learning_rate', self.lr, step = self.episode )
                    
                    self.replay( states, actions, rewards, predictions, dones, next_states )

                    state, done, score = self.env.reset(), False, 0
                    state = np.reshape( state, [ 1, self.state_size[0] ] )

                    if self.episode%100 == 0:
                        self.save_training( os.getcwd() + '/models/ppo/' )

                    break

            if self.episode >= self.EPISODES:
                break
        self.env.close()

    def run_batch(self): # train every self.Training_batch episodes
        
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score = False, 0
        
        while True:
            
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            avg = 0
            for t in range(self.Training_batch):
                
                # self.env.render()

                # Actor picks an action
                action, action_onehot, prediction = self.act(state)
                
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action)
                
                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                
                # Update current state
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                
                if done:
                    self.episode += 1
                    print("episode: {}/{}, score: {:02d}, avg: {}, time: {:03d}".format(self.episode, self.EPISODES, int(score), avg, t ), end="\r")                    
                    with self.model.train_summary_writer.as_default():
                        tf.summary.scalar( 'Workers:/score_per_episode', score, step = self.episode )
                        tf.summary.scalar( 'Workers:/learning_rate', self.lr, step = self.episode )

                    state, done, score = self.env.reset(), False, 0
                    state = np.reshape(state, [1, self.state_size[0]])
                    
            self.replay(states, actions, rewards, predictions, dones, next_states)
            avg = np.mean( rewards )
            if avg >= self.max_average: self.lr *= 0.95
            
            if self.episode%100 == 0:
                self.save_training( os.getcwd() + '/models/naluppoadv/' )
            
            if self.episode >= self.EPISODES:
                break
        
        self.env.close()  
    
    def run_multiprocesses(self, num_worker = 4):
        
        works, parent_conns, child_conns = [], [], []
        
        for idx in range( num_worker ):
            parent_conn, child_conn = Pipe()
            work = Environment( idx, child_conn, 'LunarLander-v2', self.state_size[0], self.action_size, False )
            work.start()
            works.append( work )
            parent_conns.append( parent_conn )
            child_conns.append( child_conn )

        states =        [ [] for _ in range( num_worker ) ]
        next_states =   [ [] for _ in range( num_worker ) ]
        actions =       [ [] for _ in range( num_worker ) ]
        rewards =       [ [] for _ in range( num_worker ) ]
        dones =         [ [] for _ in range( num_worker ) ]
        predictions =   [ [] for _ in range( num_worker ) ]
        score =         [  0 for _ in range( num_worker ) ]

        state = [ 0 for _ in range( num_worker ) ]
        for worker_id, parent_conn in enumerate( parent_conns ):
            state[ worker_id ] = parent_conn.recv()

        while self.episode < self.EPISODES:
            
            predictions_list = self.model.probs( np.reshape( state, [ num_worker, self.state_size[0] ] ) ).numpy()
            actions_list = [ np.random.choice( self.action_size, p = i ) for i in predictions_list ]

            for worker_id, parent_conn in enumerate( parent_conns ):
                
                parent_conn.send( actions_list[ worker_id ] )
                action_onehot = np.zeros( [ self.action_size ] )
                action_onehot[ actions_list[ worker_id ] ] = 1
                actions[ worker_id ].append( action_onehot )
                predictions[ worker_id ].append( predictions_list[ worker_id ] )

            for worker_id, parent_conn in enumerate( parent_conns ):
                
                next_state, reward, done, _ = parent_conn.recv()

                states[ worker_id ].append( state[ worker_id ] )
                next_states[ worker_id ].append( next_state )
                rewards[ worker_id ].append( reward )
                dones[ worker_id ].append( done )
                state[ worker_id ] = next_state
                score[ worker_id ] += reward

                if done:

                    print("episode: {}/{}, worker: {}, score: {}".format(self.episode, self.EPISODES, worker_id, score[worker_id]), end="\r", flush=True)

                    with self.model.train_summary_writer.as_default():
                        tf.summary.scalar( 'Workers:{}/score_per_episode'.format(worker_id), score[worker_id], step = self.episode )

                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1

            l_states =       []
            l_next_states =  []
            l_actions =      []
            l_rewards =      []
            l_dones =        []
            l_predictions =  []
            for worker_id in range(num_worker):
                
                if len(states[worker_id]) >= self.Training_batch:

                    l_states.extend( states[worker_id] )
                    l_next_states.extend( next_states[worker_id] )
                    l_actions.extend( actions[worker_id] )
                    l_rewards.extend( rewards[worker_id] )
                    l_dones.extend( dones[worker_id] )
                    l_predictions.extend( predictions[worker_id] )
                    
                    states[worker_id]      = []
                    next_states[worker_id] = []
                    actions[worker_id]     = []
                    rewards[worker_id]     = []
                    dones[worker_id]       = []
                    predictions[worker_id] = []

            if len( l_states ) > 0:
                self.replay( l_states, l_actions, l_rewards, l_predictions, l_dones, l_next_states )
                if self.replay_count%100 == 0:
                    self.save_training( os.getcwd() + '/models/naluppoadv/' )
                
        # terminating processes after while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()

    def test(self, test_episodes = 100):
        self.restore_training( os.getcwd() + '/models/naluppoadv/' )
        for e in range(100):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax( self.model.probs(state).numpy()[0] )
                state, reward, done, _ = self.env.step( action )
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, test_episodes, score))
                    break
        self.env.close()

    def save_training(self, directory):
        save_checkpoint( self.model, directory + 'local', self.replay_count )

    def restore_training(self, directory):
        self.replay_count = restore_checkpoint( self.model, directory + 'local' )
