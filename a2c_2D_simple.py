import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import Dense, Conv2D
import numpy as np

from ann_utils import dense, conv2d, softmax, flatten, categorical_sample, gather, l2_loss

class EncoderModel(Model):
    def __init__(self):
        super(EncoderModel, self).__init__()
        self.lc1 = Conv2D( 16, 8, ( 4, 4 ), padding="VALID", kernel_initializer = tf.keras.initializers.Orthogonal() )
        self.lc2 = Conv2D( 32, 4, ( 2, 2 ), padding="VALID", kernel_initializer = tf.keras.initializers.Orthogonal() )
        self.ld1 = Dense( 256, kernel_initializer = tf.keras.initializers.Orthogonal() )
    
    def call(self, state):

        c1 = self.lc1( state )
        c1 = tf.nn.relu( c1 )

        c2 = self.lc2( c1 )
        c2 = tf.nn.relu( c2 )
        
        features = flatten( c2 )

        l1 = self.lc2( features )
        l1 = tf.nn.relu( l1 )

        return l1

    def update_model(self, model):
        t = 0.25
        self.set_weights( [ t * x + ( ( 1.0 - t ) * y ) for x, y in zip ( model.get_weights(), self.get_weights() ) ] )
        
class ActorModel(Model):

    def __init__(self, num_actions):
        super(ActorModel, self).__init__()
        self.lact = Dense( num_actions, kernel_initializer = tf.keras.initializers.Orthogonal() )
    
    def call(self, features):
        logits = self.lact( features )
        act = softmax( logits )
        return logits, act

    def update_model(self, model):
        t = 0.25
        self.set_weights( [ t * x + ( ( 1.0 - t ) * y ) for x, y in zip ( model.get_weights(), self.get_weights() ) ] )

    def get_action(self, features):
        _, policy = self.__call__( features )
        action = categorical_sample( policy )
        return policy, action

class CriticModel(Model):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.lval = Dense( 1, kernel_initializer = tf.keras.initializers.Orthogonal() )
    
    def call(self, features):
        value = self.lval( features )
        return value

    def update_model(self, model):
        t = 0.25
        self.set_weights( [ t * x + ( ( 1.0 - t ) * y ) for x, y in zip ( model.get_weights(), self.get_weights() ) ] )

class PPOSGDTrain:

    def __init__(self, embeding_model, actor_model, critic_model, name,
        lr=1e-5, df=.99, en=1e-4):
        self.emodel = embeding_model
        self.amodel = actor_model
        self.cmodel = critic_model
        self.lr = lr
        self.df = df
        self.en = en

        self.e_opt = tf.keras.mixed_precision.LossScaleOptimizer( optimizers.RMSprop( lr = self.lr ) )
        self.a_opt = tf.keras.mixed_precision.LossScaleOptimizer( optimizers.RMSprop( lr = self.lr ) )
        self.c_opt = tf.keras.mixed_precision.LossScaleOptimizer( optimizers.RMSprop( lr = self.lr ) )

        self.log_dir = 'logs/{}'.format(name)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.reward_board = tf.keras.metrics.Mean('reward_board', dtype=tf.float32)
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_loss_c = tf.keras.metrics.Mean('train_loss_c', dtype=tf.float32)

    def actor_loss(self, states, actions, advantages):

        policy = self.amodel( states )

        # SparseCategoricalCrossentropy = ce loss with not one-hot encoded output
        # from_logits = True  =>  cross_entropy with soft_max
        entropy = losses.categorical_crossentropy( policy, policy, from_logits = False )
        
        ce_loss = losses.SparseCategoricalCrossentropy( from_logits = False )
        log_pi = ce_loss( actions, policy )
        
        policy_loss = log_pi * np.array( advantages )
        
        policy_loss = tf.reduce_mean( policy_loss )

        return policy_loss - self.en * entropy

    def critic_loss(self, states, rewards, dones):
        reward_sum = self.cmodel( states ) * int( not dones[-1] )        
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + self.df * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor( np.array( discounted_rewards )[:, None], dtype = tf.float32 )
        values = self.cmodel( states )
        error = tf.square( values - discounted_rewards ) * 0.5
        error = tf.reduce_mean( error )
        return error

    def compute_advantages(self, states, rewards, dones):
        reward_sum = self.amodel( states ) * int( not dones[-1] )
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + self.df * reward_sum
            discounted_rewards.append( reward_sum )
        discounted_rewards.reverse()
        values = self.amodel( states )
        advantages = discounted_rewards - values
        return advantages

    def train(self, states, actions, rewards, next_states, dones):

        states = tf.convert_to_tensor( states, dtype = tf.float32 )

        embeding_variable = self.emodel.trainable_variables

        critic_variable = self.cmodel.trainable_variables
        with tf.GradientTape() as tape_critic:
            tape_critic.watch( embeding_variable + critic_variable )
            features = self.emodel( states )
            c_loss = self.critic_loss( features, rewards, dones )
            critic_loss = self.c_opt.get_scaled_loss( c_loss )

        advantages = self.compute_advantages( features, rewards, dones )
        actor_variable = self.amodel.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch( embeding_variable + actor_variable )
            features = self.emodel( states )
            a_loss = self.actor_loss( features, actions, advantages )
            actor_loss = self.a_opt.get_scaled_loss( a_loss )

        # gradient descent will be applied automatically
        critic_grads = tape_critic.gradient( critic_loss, critic_variable + embeding_variable )
        c_grad = self.c_opt.get_unscaled_gradients( critic_grads )
        self.c_opt.apply_gradients( zip( c_grad[:len(critic_variable)], critic_variable ) )

        actor_grads = tape.gradient( actor_loss, actor_variable + embeding_variable )
        a_grad = self.a_opt.get_unscaled_gradients( actor_grads )
        self.a_opt.apply_gradients( zip( a_grad[:len(actor_variable)], actor_variable ) )

        e_grads = [ ( x + y ) for x,y in zip( c_grad[len(critic_variable):], a_grad[len(actor_variable):] ) ]
        self.e_opt.apply_gradients( zip( e_grads, embeding_variable ) )

        # loss
        self.train_loss(tf.reduce_mean(a_loss))
        self.train_loss_c(tf.reduce_mean(c_loss))
 

class A2CTrain:

    def __init__(self, embeding_model, actor_model, critic_model, name,
        lr=1e-5, df=.99, en=1e-4):
        self.emodel = embeding_model
        self.amodel = actor_model
        self.cmodel = critic_model
        self.lr = lr
        self.df = df
        self.en = en

        self.e_opt = tf.keras.mixed_precision.LossScaleOptimizer( optimizers.RMSprop( lr = self.lr ) )
        self.a_opt = tf.keras.mixed_precision.LossScaleOptimizer( optimizers.RMSprop( lr = self.lr ) )
        self.c_opt = tf.keras.mixed_precision.LossScaleOptimizer( optimizers.RMSprop( lr = self.lr ) )

        self.log_dir = 'logs/{}'.format(name)
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.reward_board = tf.keras.metrics.Mean('reward_board', dtype=tf.float32)
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_loss_c = tf.keras.metrics.Mean('train_loss_c', dtype=tf.float32)

    def actor_loss(self, states, actions, advantages):

        policy = self.amodel( states )

        # SparseCategoricalCrossentropy = ce loss with not one-hot encoded output
        # from_logits = True  =>  cross_entropy with soft_max
        entropy = losses.categorical_crossentropy( policy, policy, from_logits = False )
        
        ce_loss = losses.SparseCategoricalCrossentropy( from_logits = False )
        log_pi = ce_loss( actions, policy )
        
        policy_loss = log_pi * np.array( advantages )
        
        policy_loss = tf.reduce_mean( policy_loss )

        return policy_loss - self.en * entropy

    def critic_loss(self, states, rewards, dones):
        reward_sum = self.cmodel( states ) * int( not dones[-1] )        
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + self.df * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor( np.array( discounted_rewards )[:, None], dtype = tf.float32 )
        values = self.cmodel( states )
        error = tf.square( values - discounted_rewards ) * 0.5
        error = tf.reduce_mean( error )
        return error

    def compute_advantages(self, states, rewards, dones):
        reward_sum = self.amodel( states ) * int( not dones[-1] )
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + self.df * reward_sum
            discounted_rewards.append( reward_sum )
        discounted_rewards.reverse()
        values = self.amodel( states )
        advantages = discounted_rewards - values
        return advantages

    def train(self, states, actions, rewards, next_states, dones):

        states = tf.convert_to_tensor( states, dtype = tf.float32 )

        embeding_variable = self.emodel.trainable_variables

        critic_variable = self.cmodel.trainable_variables
        with tf.GradientTape() as tape_critic:
            tape_critic.watch( embeding_variable + critic_variable )
            features = self.emodel( states )
            c_loss = self.critic_loss( features, rewards, dones )
            critic_loss = self.c_opt.get_scaled_loss( c_loss )

        advantages = self.compute_advantages( features, rewards, dones )
        actor_variable = self.amodel.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch( embeding_variable + actor_variable )
            features = self.emodel( states )
            a_loss = self.actor_loss( features, actions, advantages )
            actor_loss = self.a_opt.get_scaled_loss( a_loss )

        # gradient descent will be applied automatically
        critic_grads = tape_critic.gradient( critic_loss, critic_variable + embeding_variable )
        c_grad = self.c_opt.get_unscaled_gradients( critic_grads )
        self.c_opt.apply_gradients( zip( c_grad[:len(critic_variable)], critic_variable ) )

        actor_grads = tape.gradient( actor_loss, actor_variable + embeding_variable )
        a_grad = self.a_opt.get_unscaled_gradients( actor_grads )
        self.a_opt.apply_gradients( zip( a_grad[:len(actor_variable)], actor_variable ) )

        e_grads = [ ( x + y ) for x,y in zip( c_grad[len(critic_variable):], a_grad[len(actor_variable):] ) ]
        self.e_opt.apply_gradients( zip( e_grads, embeding_variable ) )

        # loss
        self.train_loss(tf.reduce_mean(a_loss))
        self.train_loss_c(tf.reduce_mean(c_loss))

class A2CGaeTrain:

    def __init__(self, embeding_model, actor_model, critic_model, name,
        lr=1e-6, gamma=.99, en=1e-6, tau=1):
        self.emodel = embeding_model
        self.amodel = actor_model
        self.cmodel = critic_model
        self.lr = lr
        self.epsilon = 1e-8
        self.en = en
        self.gamma = gamma
        self.tau = tau

        self.e_opt = tf.keras.mixed_precision.LossScaleOptimizer( optimizers.RMSprop( lr = self.lr ) )
        self.a_opt = tf.keras.mixed_precision.LossScaleOptimizer( optimizers.RMSprop( lr = self.lr ) )
        self.c_opt = tf.keras.mixed_precision.LossScaleOptimizer( optimizers.RMSprop( lr = self.lr ) )

    def actor_loss(self, states, old_log_policies, actions, advantages):

        policy = self.amodel( states )

        # SparseCategoricalCrossentropy = ce loss with not one-hot encoded output
        # from_logits = True  =>  cross_entropy with soft_max
        entropy = losses.categorical_crossentropy( policy, policy, from_logits = False )
        
        ## log da ação selecionada na softmax
        new_log_policy = tf.math.log( gather( policy, actions ) )
        
        ratio = tf.exp( new_log_policy - old_log_policies )

        ce_loss = losses.SparseCategoricalCrossentropy( from_logits = False )
        log_pi = ce_loss( actions, policy )
        policy_loss = log_pi * np.asarray( advantages )

        #advantages = np.asarray( advantages )
        #policy_loss = -tf.reduce_mean( tf.minimum( ratio * advantages, tf.clip_by_value( ratio, 1.0 - self.epsilon, 1.0 + self.epsilon ) * advantages ) )
        #policy_loss = -tf.reduce_mean( policy_loss )

        return policy_loss - self.en * entropy, policy

    def critic_loss(self, states, rewards):
        values = self.cmodel( states )
        return tf.keras.losses.huber( rewards, values )
    
    def get_gradients(self, states, actions, rewards, next_states, values, old_log_policies, R, adv, dones):

        states = tf.convert_to_tensor( states, dtype = tf.float32 )

        features = self.emodel( states[-1][None, ...] )

        embeding_variable = self.emodel.trainable_variables

        critic_variable = self.cmodel.trainable_variables
        with tf.GradientTape() as tape_critic:
            tape_critic.watch( embeding_variable + critic_variable )
            features = self.emodel( states )
            c_loss = self.critic_loss( features, tf.convert_to_tensor( R[:,np.newaxis], dtype = tf.float32 ) )
            critic_loss = self.c_opt.get_scaled_loss( 0.5 * c_loss + l2_loss( embeding_variable + critic_variable, 1e-4 ) )

        # gradient descent will be applied automatically
        critic_grads = tape_critic.gradient( critic_loss, critic_variable + embeding_variable )
        c_grad = self.c_opt.get_unscaled_gradients( critic_grads )

        actor_variable = self.amodel.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch( embeding_variable + actor_variable )
            features = self.emodel( states )
            a_loss, policy = self.actor_loss( features, old_log_policies, actions, tf.convert_to_tensor( adv[:,np.newaxis], dtype = tf.float32 ) )
            actor_loss = self.a_opt.get_scaled_loss( a_loss + l2_loss( embeding_variable + actor_variable, 1e-4 ) )

        actor_grads = tape.gradient( actor_loss, actor_variable + embeding_variable )
        a_grad = self.a_opt.get_unscaled_gradients( actor_grads )

        e_grads = [ x + y for x,y in zip( c_grad[len(critic_variable):], a_grad[len(actor_variable):] ) ]

        return tf.reduce_mean(a_loss), tf.reduce_mean(c_loss), c_grad[:len(critic_variable)], a_grad[:len(actor_variable)], e_grads, tf.reduce_mean(policy)
    
    def apply_grads(self, c_grads, a_grads, e_grads):

        critic_variable = self.cmodel.trainable_variables
        actor_variable = self.amodel.trainable_variables
        embeding_variable = self.emodel.trainable_variables

        c_grads_clipped = [ tf.clip_by_value( grad, -1., 1.) for grad in c_grads ]
        a_grads_clipped = [ tf.clip_by_value( grad, -1., 1.) for grad in a_grads ]
        e_grads_clipped = [ tf.clip_by_value( grad, -1., 1.) for grad in e_grads ]

        self.c_opt.apply_gradients( zip( c_grads_clipped, critic_variable ) )
        self.a_opt.apply_gradients( zip( a_grads_clipped, actor_variable ) )
        self.e_opt.apply_gradients( zip( e_grads_clipped, embeding_variable ) )


