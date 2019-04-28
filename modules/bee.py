import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.processing import ProcessingModule
from modules.goal_predicting import GoalPredictingProcessingModule
from modules.action import ActionModule
from modules.word_counting import WordCountingModule
from modules.agent import AgentModule

""" Swarm action defining"""

class BeeModule(AgentModule):

    def __init__(self, config):
        super().__init__(self, config)
        self.swarm_action_processor = SwarmActModule(config.swarm_action_processor)
        self.hive_num = config.hive_num
    
    def swarm_get_action(self, game, agent, utterance_feat):
        """
        need figure out new mem part
        """
        vote, new_mem = self.swarm_action_processor(utterance_feat, game.memories["action"][:, agent], 
        self.training)
        self.update_mem(game, "action", new_mem, agent)
        votes[:, agent, :] = vote
    
    def forward(self, game):
        timesteps = []
        for t in range(self.time_horizon):
            votes = self.Tensor(game.batch_size, game.num_agents,
                                self.hive_num, require_grad=True)
            utterances = self.Tensor(game.batch_size, game.num_agents,
                                     self.vocab_size, require_grad=True)
            for agent in range(game.num_agents):
                
                
