
class ExplorerWorker:

    def __init__(self, name, actions):

        self.e_model = EncoderModel()
        self.act_model = ActorModel( actions )
        self.c_model = CriticModel()
        self.name = name
        
        self.log_dir = 'logs/{}'.format(name)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.boards = {
            'reward': tf.keras.metrics.Mean('reward_board', dtype=tf.float32)
        }

    def set_board(self, name, value):
        self.boards[name](value)
    
    def get_board(self, name):
        self.boards[name].result()

    def get_values(self, state):

        embeding = self.e_model( tf.convert_to_tensor( state[None, ...], dtype = tf.float32 ) )
        action, p, log_policy = self.act_model.get_action( embeding )
        value = np.asarray( self.c_model( embeding ) )[0][0]

        return action, p, log_policy, value

class TotalWorker:

    def __init__(self, name, actions):

        self.e_model = EncoderModel()
        self.act_model = ActorModel( actions )
        self.c_model = CriticModel()
        self.t_model = A2CGaeTrain(self.e_model, self.act_model, self.c_model, name)
        self.name = name
        
        self.log_dir = 'logs/{}'.format(name)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.boards = {
            'reward': tf.keras.metrics.Mean('reward_board', dtype=tf.float32),
            'actor_loss': tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
            'critic_loss': tf.keras.metrics.Mean('train_loss_c', dtype=tf.float32)
        }

    def set_board(self, name, value):
        self.boards[name](value)
    
    def get_board(self, name):
        self.boards[name].result()

    def get_values(self, state):
        embeding = self.e_model( tf.convert_to_tensor( state[None, ...], dtype = tf.float32 ) )
        action, p, log_policy = self.act_model.get_action( embeding )
        value = np.asarray( self.c_model( embeding ) )[0][0]
        return action, p, log_policy, value

    def train(self, states, actions, rewards, _states, values, old_log_policies, dones):
        return self.t_model.train( states, actions, rewards, _states, values, old_log_policies, dones )

class Worker(Thread):

    def __init__(self, model, env, num_local_steps):

        threading.Thread.__init__(self)
        self.model = model
        self.queue = Queue(5)        
        self.num_local_steps = num_local_steps
        self.env = env

    def start_runner(self):
        self.start()

    def run(self):
        self._run()

    def _run(self):

        self.model.steps = 0

        if type(self.model) is ExplorerWorker:

            rollout_provider = ac_env_runner_explorer( self.model, self.env, self.num_local_steps )            
            while True: self.queue.put( next( rollout_provider ), timeout = 600.0 )            
        
        else:
            while True: ac_env_runner_eval( self.env, self.model, self.num_local_steps )

class Trainer(Thread):

    def __init__(self, worker):

        threading.Thread.__init__(self)
        self.worker = worker

    def start_runner(self):
        self.start()

    def run(self):
        self._run()

    def _run(self):

        while True:
            states, _state, log_policy, action, value, reward, done = self._process()
            # values to train
            # update models

    def _pull_batch_from_queue(self):
        """
        Take a rollout from the queue of the thread runner.
        """
        rollout = self.worker.queue.get( timeout = 600.0 )
        while not rollout.terminal:
            try:
                vls = self.worker.queue.get_nowait()
                rollout.extend( vls )
            except:
                break
        print(rollout.size())
        return rollout

    def _process(self):
        
        rollout = self._pull_batch_from_queue()
        
        states = rollout.states
        _states = rollout._states
        log_policy = rollout.log_polices
        action = rollout.actions
        value = rollout.values
        reward = rollout.rewards
        done = rollout.dones

        return states, _states, log_policy, action, value, reward, done
