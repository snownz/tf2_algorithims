import tensorflow as tf
from custom_rl import SimplePPONetwork, NaluPPONetwork, NaluAdavancedPPONetwork, NaluAdavanced2PPONetwork, IQNNaluPPONetwork
from custom_rl import SimpleRNNPPONetwork
import copy
import numpy as np
from utils import save_checkpoint, restore_checkpoint
from multi_env_warper import Environment
from multiprocessing import Pipe
import os

class PPOAgentBase():

    def __init__(self, env, name, env_name, epochs):

        self.env = env
        self.env_name = env_name
        self.name = name
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.replay_count = 0
        self.epochs = epochs
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        self.model = object
        self.get_probs = object

    def act(self, state):
        
        prediction = self.get_probs( state )
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
        self.model.train( states, values, target, advantages, predictions, actions, rewards, epochs = self.epochs )

        with self.model.train_summary_writer.as_default():
            
            tf.summary.scalar( 'Data/actor_loss_per_replay', self.model.a_loss.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/critic_loss_per_replay', self.model.c_loss.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/reward_per_replay', self.model.reward.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/learning_rate_per_replay', self.model.lr( self.replay_count * self.epochs ).numpy(), step = self.replay_count )

        self.replay_count += 1

    def test(self, test_episodes=100, load=True):
        
        if load:
            self.restore_training( os.getcwd() + '/models/{}/{}'.format( self.name, self.env_name ) )
        
        scores = []
        for e in range(test_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax( self.get_probs( state ) )
                state, reward, done, _ = self.env.step( action )
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, test_episodes, score))
                    scores.append( score )
                    break
        print("Final Score: {}".format( np.mean( score ) ) )        
        if load:
            self.env.close()

    def save_training(self, directory):
        save_checkpoint( self.model, directory, self.replay_count )

    def restore_training(self, directory):
        self.replay_count = restore_checkpoint( self.model, directory )

class PPOAgentRnnBase():

    def __init__(self, env, name, env_name, epochs):

        self.env = env
        self.env_name = env_name
        self.name = name
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.replay_count = 0
        self.epochs = epochs
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        self.model = object
        self.get_probs = object

    def act(self, state):
        
        prediction = self.get_probs( state )
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
        
        b_states, b_values, b_target, b_advantages, b_predictions, b_actions, b_rewards = [],[],[],[],[],[],[]
        for s, s_, a, r, p, d in zip( states, next_states, actions, rewards, predictions, dones ):
            
            v_states = np.expand_dims( np.array( s ).squeeze(1), 0 )
            v_next_states = np.expand_dims( np.array( s_ ).squeeze(1), 0 )

            # Get Critic network predictions 
            values, _ = self.model.value( v_states, self.model.zero_states(1) )
            next_values, _ = self.model.value( v_next_states, self.model.zero_states(1) )

            values = values.numpy()[0]
            next_values = next_values.numpy()[0]

            # Compute discounted rewards and advantages
            advantages, target = self.get_gaes( r, d, np.squeeze( values ), np.squeeze( next_values ) )

            div = len(s) // self.max_time
            
            for dv in range(div):
                
                start = ( dv * self.max_time )
                end = start + self.max_time
                
                b_states.append( np.expand_dims( np.array( s[start:end] ).squeeze(1), 0 ).astype(np.float32) )
                b_values.append( np.expand_dims( np.array( values[start:end] ), 0 ).astype(np.float32) )
                b_target.append( np.expand_dims( np.array( target[start:end] ), 0 ).astype(np.float32) )
                b_advantages.append( np.expand_dims( np.array( advantages[start:end] ), 0 ).astype(np.float32) )
                b_predictions.append( np.expand_dims( np.array( p[start:end] ), 0 ).astype(np.float32) )
                b_actions.append( np.expand_dims( np.array( a[start:end] ), 0 ).astype(np.float32) )
                b_rewards.append( np.expand_dims( np.array( r[start:end] ), 0 ).astype(np.float32) )
        
        b_states = tf.concat( b_states, axis = 0 )
        b_values = tf.concat( b_values, axis = 0 )
        b_target = tf.concat( b_target, axis = 0 )
        b_advantages = tf.concat( b_advantages, axis = 0 )
        b_predictions = tf.concat( b_predictions, axis = 0 )
        b_actions = tf.concat( b_actions, axis = 0 )
        b_rewards = tf.concat( b_rewards, axis = 0 )

        # training Actor and Critic networks
        self.model.train( b_states, b_values, b_target, b_advantages, b_predictions, b_actions, b_rewards, epochs = self.epochs )

        with self.model.train_summary_writer.as_default():
            
            tf.summary.scalar( 'Data/actor_loss_per_replay', self.model.a_loss.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/critic_loss_per_replay', self.model.c_loss.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/reward_per_replay', self.model.reward.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/learning_rate_per_replay', self.model.lr( self.replay_count * self.epochs ).numpy(), step = self.replay_count )

        self.replay_count += 1

    def test(self, test_episodes=100, load=True, disable_recurrence=False):
        
        if load:
            self.restore_training( os.getcwd() + '/models/{}/{}'.format( self.name, self.env_name ) )
        
        scores = []
        for e in range(test_episodes):
            a_s = self.model.zero_states(1)
            state = self.env.reset()
            state = np.reshape(state, [1, 1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action, a_s_ = self.get_probs( state, a_s )
                action = np.argmax( action.numpy()[0] )
                
                if disable_recurrence: a_s = self.model.zero_states(1)
                else: a_s = a_s_
                
                state, reward, done, _ = self.env.step( action )
                state = np.reshape(state, [1, 1, self.state_size[0]])
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, test_episodes, score))
                    scores.append( score )
                    break
        print("Final Score: {}".format( np.mean( score ) ) )        
        if load:
            self.env.close()

    def save_training(self, directory):
        save_checkpoint( self.model, directory, self.replay_count )

    def restore_training(self, directory):
        self.replay_count = restore_checkpoint( self.model, directory )


class SimplePPOAgent(PPOAgentBase):

    def __init__(self, env, env_name):
        
        super(SimplePPOAgent, self).__init__( env, 'ppo', env_name, 10 )

        self.EPISODES = 20000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 100 # when average score is above 0 model will be saved
        self.max_steps = 200 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.Training_batch = 128

        # Create Actor-Critic network models
        self.model = SimplePPONetwork( self.action_size, env_name + "_", self.lr )
        self.model( tf.zeros( [ 1, self.state_size[0] ] ) )
        self.get_probs = lambda x: self.model.probs( x ).numpy()[ 0 ]
  
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
            work = Environment( idx, child_conn, self.env_name, self.state_size[0], self.action_size, False )
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

        test = True
        while self.episode < self.EPISODES:

            if test:
                self.test(5, load=False)
                test = False
            
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

                    print("episode: {}/{}, worker: {}, score: {}".format(self.episode, self.EPISODES, worker_id, score[worker_id]), end="\r")
                    with self.model.train_summary_writer.as_default():
                        tf.summary.scalar( 'Workers:{}/score_per_episode'.format(worker_id), score[worker_id], step = self.episode )

                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1
                    
                    if self.episode%1000 == 0:
                        test = True
                        
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
                    self.save_training( os.getcwd() + '/models/{}/{}'.format( self.name, self.env_name ) )

        # terminating processes after while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()

class SimpleRnnPPOAgent(PPOAgentRnnBase):

    def __init__(self, env, env_name, typer, mode, max_time, bs):
        
        super(SimpleRnnPPOAgent, self).__init__( env, 'rnn_{}_ppo'.format(typer), env_name, 10 )

        self.EPISODES = 20000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 100 # when average score is above 0 model will be saved
        self.max_steps = 200 # when average score is above 0 model will be saved
        self.max_time = max_time
        self.lr = 0.00025
        self.Training_batch = bs

        # Create Actor-Critic network models
        self.model = SimpleRNNPPONetwork( self.action_size, env_name + "_", self.max_time, self.lr, typer, mode )
        self.model( tf.zeros( [ 1, self.max_time, self.state_size[0] ] ), self.model.zero_states(1), self.model.zero_states(1) )
        self.get_probs = lambda x, h: self.model.probs( x, h )
  
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
        
        r_state_worker_mask = []
        for idx in range( num_worker ):
            parent_conn, child_conn = Pipe()
            work = Environment( idx, child_conn, self.env_name, self.state_size[0], self.action_size, False )
            work.start()
            works.append( work )
            parent_conns.append( parent_conn )
            child_conns.append( child_conn )

            msk = np.ones( [ num_worker, 256 ] )
            msk[idx] -= 1
            r_state_worker_mask.append( msk )

        states =        [ [] for _ in range( num_worker ) ]
        next_states =   [ [] for _ in range( num_worker ) ]
        actions =       [ [] for _ in range( num_worker ) ]
        rewards =       [ [] for _ in range( num_worker ) ]
        dones =         [ [] for _ in range( num_worker ) ]
        predictions =   [ [] for _ in range( num_worker ) ]
        score =         [  0 for _ in range( num_worker ) ]

        state = [ 0 for _ in range( num_worker ) ]
        a_state = tf.reshape( [ self.model.zero_states(1) for _ in range( num_worker ) ], [ num_worker, -1 ] )
        for worker_id, parent_conn in enumerate( parent_conns ):
            state[ worker_id ] = parent_conn.recv()

        test = True
        while self.episode < self.EPISODES:

            if test:
                self.test(5, load=False)
                test = False
            
            predictions_list, a_state_ = self.model.probs( tf.reshape( state, [ num_worker, 1, self.state_size[0] ] ),  a_state )
            
            predictions_list = predictions_list.numpy()[:,0,:]
            actions_list = [ np.random.choice( self.action_size, p = i ) for i in predictions_list ]

            a_state = a_state_

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
                    
                    # reset state for worker
                    a_state = ( r_state_worker_mask[ worker_id ] * a_state )
                    
                    print("episode: {}/{}, worker: {}, score: {}".format(self.episode, self.EPISODES, worker_id, score[worker_id]), end="\r")
                    with self.model.train_summary_writer.as_default():
                        tf.summary.scalar( 'Workers:{}/score_per_episode'.format(worker_id), score[worker_id], step = self.episode )

                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1
                    
                    if self.episode%1000 == 0:
                        test = True
                        
            l_states =       []
            l_next_states =  []
            l_actions =      []
            l_rewards =      []
            l_dones =        []
            l_predictions =  []
            for worker_id in range(num_worker):
                
                if len(states[worker_id]) >= self.Training_batch:

                    l_states.append( states[worker_id] )
                    l_next_states.append( next_states[worker_id] )
                    l_actions.append( actions[worker_id] )
                    l_rewards.append( rewards[worker_id] )
                    l_dones.append( dones[worker_id] )
                    l_predictions.append( predictions[worker_id] )
                    
                    states[worker_id]      = []
                    next_states[worker_id] = []
                    actions[worker_id]     = []
                    rewards[worker_id]     = []
                    dones[worker_id]       = []
                    predictions[worker_id] = []

            if len( l_states ) > 0:
                self.replay( l_states, l_actions, l_rewards, l_predictions, l_dones, l_next_states )
                if self.replay_count%100 == 0:
                    self.save_training( os.getcwd() + '/models/{}/{}'.format( self.name, self.env_name ) )

        # terminating processes after while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()


class NaluPPOAgent(PPOAgentBase):

    def __init__(self, env, env_name):

        super(NaluPPOAgent, self).__init__( env, 'nalu_ppo', env_name, 10 )

        self.EPISODES = 10000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 100 # when average score is above 0 model will be saved
        self.max_steps = 200 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.Training_batch = 128

        # Create Actor-Critic network models
        self.model = NaluPPONetwork( self.action_size, env_name + "_", self.lr )
        self.model( tf.zeros( [ 1, self.state_size[0] ] ) )

        self.get_probs = lambda x: self.model.probs( x ).numpy()[ 0 ]

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
            work = Environment( idx, child_conn, self.env_name, self.state_size[0], self.action_size, False )
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

        test = True
        while self.episode < self.EPISODES:

            if test:
                self.test(5, load=False)
                test = False
            
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

                    print("episode: {}/{}, worker: {}, score: {}".format(self.episode, self.EPISODES, worker_id, score[worker_id]), end="\r")
                    with self.model.train_summary_writer.as_default():
                        tf.summary.scalar( 'Workers:{}/score_per_episode'.format(worker_id), score[worker_id], step = self.episode )

                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1
                    
                    if self.episode%1000 == 0:
                        test = True
                        
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
                    self.save_training( os.getcwd() + '/models/{}/{}'.format( self.name, self.env_name ) )

        # terminating processes after while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()


class NaluAdvancedPPOAgent(PPOAgentBase):

    def __init__(self, env, env_name):

        super(NaluAdvancedPPOAgent, self).__init__( env, 'nalu_ppo_adv', env_name, 10 )

        self.EPISODES = 10000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 100 # when average score is above 0 model will be saved
        self.max_steps = 200 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.Training_batch = 1000

        # Create Actor-Critic network models
        self.model = NaluAdavancedPPONetwork( self.action_size, env_name + "_", self.lr )
        self.model( tf.zeros( [ 1, self.state_size[0] ] ) )

        self.get_probs = lambda x: self.model.probs( x ).numpy()[ 0 ]

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
            work = Environment( idx, child_conn, self.env_name, self.state_size[0], self.action_size, False )
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

        test = True
        while self.episode < self.EPISODES:

            if test:
                self.test(20, load=False)
                test = False
            
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

                    print("episode: {}/{}, worker: {}, score: {}".format(self.episode, self.EPISODES, worker_id, score[worker_id]), end="\r")
                    with self.model.train_summary_writer.as_default():
                        tf.summary.scalar( 'Workers:{}/score_per_episode'.format(worker_id), score[worker_id], step = self.episode )

                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1

                    if self.episode%1000 == 0:
                        test = True

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
                    self.save_training( os.getcwd() + '/models/{}/{}'.format( self.name, self.env_name ) )
                
        # terminating processes after while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()


class PPOAgentBaseAdv(PPOAgentBase):

    def __init__(self, env, name, env_name, epochs):
        super(PPOAgentBaseAdv, self).__init__( env, name, env_name, epochs )
    
    def act(self, state):
        
        p, l = self.model.probs( state )
        prediction = p.numpy()[0]
        lg = l.numpy()[0]
        action = np.random.choice( self.action_size, p = prediction )
        action_onehot = np.zeros( [ self.action_size ] )
        action_onehot[ action ] = 1
        return action, action_onehot, prediction, lg

    def replay(self, states, actions, rewards, predictions, dones, next_states, acs):
        
        states = np.vstack( states )
        next_states = np.vstack( next_states )
        actions = np.vstack( actions )
        acs = np.vstack( acs )
        predictions = np.vstack( predictions )

        # Get Critic network predictions 
        values, _ = self.model.value( states )
        next_values, _ = self.model.value( next_states )

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes( rewards, dones, np.squeeze( values ), np.squeeze( next_values ) )
        
        # training Actor and Critic networks
        self.model.train( states, values, target, advantages, predictions, actions, rewards, self.epochs, acs )

        with self.model.train_summary_writer.as_default():
            
            tf.summary.scalar( 'Data/actor_loss_per_replay', self.model.a_loss.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/critic_loss_per_replay', self.model.c_loss.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/reward_per_replay', self.model.reward.result(), step = self.replay_count )
            tf.summary.scalar( 'Data/learning_rate_per_replay', self.model.lr( self.replay_count * self.epochs ).numpy(), step = self.episode )


        self.replay_count += 1


class NaluAdvanced2PPOAgent(PPOAgentBaseAdv):

    def __init__(self, env, env_name):

        super(NaluAdvanced2PPOAgent, self).__init__( env, 'nalu_ppo_adv2', env_name, 10 )

        self.EPISODES = 10000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 100 # when average score is above 0 model will be saved
        self.max_steps = 200 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.Training_batch = 1000

        # Create Actor-Critic network models
        self.model = NaluAdavanced2PPONetwork( self.action_size, env_name + "_", self.lr )
        self.model( tf.zeros( [ 1, self.state_size[0] ] ) )

        self.get_probs = lambda x: self.model.probs( x )[0].numpy()[ 0 ]

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
            work = Environment( idx, child_conn, self.env_name, self.state_size[0], self.action_size, False )
            work.start()
            works.append( work )
            parent_conns.append( parent_conn )
            child_conns.append( child_conn )

        states =        [ [] for _ in range( num_worker ) ]
        next_states =   [ [] for _ in range( num_worker ) ]
        actions =       [ [] for _ in range( num_worker ) ]
        acs =           [ [] for _ in range( num_worker ) ]
        rewards =       [ [] for _ in range( num_worker ) ]
        dones =         [ [] for _ in range( num_worker ) ]
        predictions =   [ [] for _ in range( num_worker ) ]
        score =         [  0 for _ in range( num_worker ) ]

        state = [ 0 for _ in range( num_worker ) ]
        for worker_id, parent_conn in enumerate( parent_conns ):
            state[ worker_id ] = parent_conn.recv()

        test = True
        while self.episode < self.EPISODES:

            if test:
                self.test(20, load=False)
                test = False
            
            predictions_list = self.model.probs( np.reshape( state, [ num_worker, self.state_size[0] ] ) )
            actions_list = [ np.random.choice( self.action_size, p = i ) for i in predictions_list[0].numpy() ]
            acs_list = [ i for i in predictions_list[1].numpy() ]

            for worker_id, parent_conn in enumerate( parent_conns ):
                
                parent_conn.send( actions_list[ worker_id ] )
                action_onehot = np.zeros( [ self.action_size ] )
                action_onehot[ actions_list[ worker_id ] ] = 1
                actions[ worker_id ].append( action_onehot )
                acs[ worker_id ].append( acs_list[ worker_id ] )
                predictions[ worker_id ].append( predictions_list[0][ worker_id ] )

            for worker_id, parent_conn in enumerate( parent_conns ):
                
                next_state, reward, done, _ = parent_conn.recv()

                states[ worker_id ].append( state[ worker_id ] )
                next_states[ worker_id ].append( next_state )
                rewards[ worker_id ].append( reward )
                dones[ worker_id ].append( done )
                state[ worker_id ] = next_state
                score[ worker_id ] += reward

                if done:

                    print("episode: {}/{}, worker: {}, score: {}".format(self.episode, self.EPISODES, worker_id, score[worker_id]), end="\r")
                    with self.model.train_summary_writer.as_default():
                        tf.summary.scalar( 'Workers:{}/score_per_episode'.format(worker_id), score[worker_id], step = self.episode )

                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1

                    if self.episode%1000 == 0:
                        test = True

            l_states =       []
            l_next_states =  []
            l_actions =      []
            l_acs =          []
            l_rewards =      []
            l_dones =        []
            l_predictions =  []
            for worker_id in range(num_worker):
                
                if len(states[worker_id]) >= self.Training_batch:

                    l_states.extend( states[worker_id] )
                    l_next_states.extend( next_states[worker_id] )
                    l_actions.extend( actions[worker_id] )
                    l_acs.extend( acs[worker_id] )
                    l_rewards.extend( rewards[worker_id] )
                    l_dones.extend( dones[worker_id] )
                    l_predictions.extend( predictions[worker_id] )
                    
                    states[worker_id]      = []
                    next_states[worker_id] = []
                    actions[worker_id]     = []
                    acs[worker_id]     = []
                    rewards[worker_id]     = []
                    dones[worker_id]       = []
                    predictions[worker_id] = []

            if len( l_states ) > 0:
                self.replay( l_states, l_actions, l_rewards, l_predictions, l_dones, l_next_states, l_acs )
                if self.replay_count%100 == 0:
                    self.save_training( os.getcwd() + '/models/{}/{}'.format( self.name, self.env_name ) )
                
        # terminating processes after while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()


class IQNNaluPPOAgent(PPOAgentBaseAdv):

    def __init__(self, env, env_name):

        super(IQNNaluPPOAgent, self).__init__( env, 'iqn_nalu_ppo', env_name, 10 )

        self.EPISODES = 30000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 100 # when average score is above 0 model will be saved
        self.max_steps = 200 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.Training_batch = 1000

        # Create Actor-Critic network models
        self.model = IQNNaluPPONetwork( env_name + "_",  self.action_size, 64, 8, self.lr )
        self.model( tf.zeros( [ 1, self.state_size[0] ] ) )

        self.get_probs = lambda x: self.model.probs( x )[0].numpy()[ 0 ]

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
            work = Environment( idx, child_conn, self.env_name, self.state_size[0], self.action_size, False )
            work.start()
            works.append( work )
            parent_conns.append( parent_conn )
            child_conns.append( child_conn )

        states =        [ [] for _ in range( num_worker ) ]
        next_states =   [ [] for _ in range( num_worker ) ]
        actions =       [ [] for _ in range( num_worker ) ]
        acs =           [ [] for _ in range( num_worker ) ]
        rewards =       [ [] for _ in range( num_worker ) ]
        dones =         [ [] for _ in range( num_worker ) ]
        predictions =   [ [] for _ in range( num_worker ) ]
        score =         [  0 for _ in range( num_worker ) ]

        state = [ 0 for _ in range( num_worker ) ]
        for worker_id, parent_conn in enumerate( parent_conns ):
            state[ worker_id ] = parent_conn.recv()

        test = True
        while self.episode < self.EPISODES:

            if test:
                self.test(5, load=False)
                test = False
            
            predictions_list = self.model.probs( np.reshape( state, [ num_worker, self.state_size[0] ] ) )
            actions_list = [ np.random.choice( self.action_size, p = i ) for i in predictions_list[0].numpy() ]
            acs_list = [ i for i in predictions_list[1].numpy() ]

            for worker_id, parent_conn in enumerate( parent_conns ):
                
                parent_conn.send( actions_list[ worker_id ] )
                action_onehot = np.zeros( [ self.action_size ] )
                action_onehot[ actions_list[ worker_id ] ] = 1
                actions[ worker_id ].append( action_onehot )
                acs[ worker_id ].append( acs_list[ worker_id ] )
                predictions[ worker_id ].append( predictions_list[0][ worker_id ] )

            for worker_id, parent_conn in enumerate( parent_conns ):
                
                next_state, reward, done, _ = parent_conn.recv()

                states[ worker_id ].append( state[ worker_id ] )
                next_states[ worker_id ].append( next_state )
                rewards[ worker_id ].append( reward )
                dones[ worker_id ].append( done )
                state[ worker_id ] = next_state
                score[ worker_id ] += reward

                if done:

                    print("episode: {}/{}, worker: {}, score: {}".format(self.episode, self.EPISODES, worker_id, score[worker_id]), end="\r")
                    with self.model.train_summary_writer.as_default():
                        tf.summary.scalar( 'Workers:{}/score_per_episode'.format(worker_id), score[worker_id], step = self.episode )

                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1

                    if self.episode%1000 == 0:
                        test = True

            l_states =       []
            l_next_states =  []
            l_actions =      []
            l_acs =          []
            l_rewards =      []
            l_dones =        []
            l_predictions =  []
            for worker_id in range(num_worker):
                
                if len(states[worker_id]) >= self.Training_batch:

                    l_states.extend( states[worker_id] )
                    l_next_states.extend( next_states[worker_id] )
                    l_actions.extend( actions[worker_id] )
                    l_acs.extend( acs[worker_id] )
                    l_rewards.extend( rewards[worker_id] )
                    l_dones.extend( dones[worker_id] )
                    l_predictions.extend( predictions[worker_id] )
                    
                    states[worker_id]      = []
                    next_states[worker_id] = []
                    actions[worker_id]     = []
                    acs[worker_id]     = []
                    rewards[worker_id]     = []
                    dones[worker_id]       = []
                    predictions[worker_id] = []

            if len( l_states ) > 0:
                self.replay( l_states, l_actions, l_rewards, l_predictions, l_dones, l_next_states, l_acs )
                if self.replay_count%100 == 0:
                    self.save_training( os.getcwd() + '/models/{}/{}'.format( self.name, self.env_name ) )
                
        # terminating processes after while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()
