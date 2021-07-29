class Config:

    def __init__(self):
        self.BUFFER_SIZE = int(2e4)  # replay buffer size
        self.BATCH_SIZE = 256        # minibatch size
        self.GAMMA = 0.995           # discount factor
        self.TAU = 1e-3              # for soft update of target parameters
        self.LR_ACTOR = 1e-3         # learning rate of the actor
        self.LR_CRITIC = 1e-3        # learning rate of the critic

        #Noise params
        self.ou_mu = 0.0  
        self.ou_theta = 0.15  
        self.ou_sigma = 0.20  
        self.EPS_START = 1.0        # Initial epsilon value (explore) 
        self.EPS_END = 0.1          # Last epsilon value  (exploit)

        self.EXP_STEPS = 3e4
        self.LIN_EPS_DECAY = self.EPS_START/self.EXP_STEPS  
        self.LEARN_TIMES = 4
        self.UPDATE_EVERY = 2