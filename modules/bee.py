import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.processing import ProcessingModule
from modules.goal_predicting import GoalPredictingProcessingModule
from modules.action import ActionModule
from modules.word_counting import WordCountingModule
from modules.agent import AgentModule
from .swarm_action import SwarmActModule
from .scout_action import ScoutActModule

""" Swarm action defining"""

class BeeModule(AgentModule):

    def __init__(self, config):
        super().__init__(self, config)
        # config need to be added, need to create scoutActmodule
        self.swarm_action_processor = SwarmActModule(config.swarm_action_processor)
        self.hive_num = config.hive_num
        self.scouts_action_processor = ScoutActModule(config.scout_action_processor)
    
    def swarm_get_action(self, game, agent, utterance_feat, votes):
        """
        need figure out new mem part
        """
        vote, new_mem = self.swarm_action_processor(utterance_feat, game.memories["action"][:, agent], 
        self.training)
        self.update_mem(game, "action", new_mem, agent)
        votes[:, agent, :] = vote

    def scouts_get_action(self,game, agent, physical_feat, utterance_feat,
                         movements, utterances, votes):
        vote, movement, utterance, new_men = self.scout_action_processor(physical_feat, utterance_feat,
                                             game.memories["action"][:, agent],
                                             self.training)
        self.update_mem(game, "action", new_mem, agent)
        movements[:, agent, :] = movement
        votes[:, agent, :] = vote
        utterances[:, agent, :] = utterances

    def get_utterance_feat(self, game, agent):
        utterance_processes = self.Tensor(game.batch_size, game.num_agents,
                              self.processing_hidden_size, require_grad=True)
        for other_agent in range(game.num_agents):
            self.process_utterances(game, agent, other_agent,
                                    utterance_processes, goal_predictions)
            return self.utterance_pooling(utterance_processes)
    
    

    def forward(self, game):
        timesteps = []
        utters = []
        utters_nums_t = []
        for t in range(self.time_horizon):
            votes = self.Tensor(game.batch_size, game.num_agents,
                                self.hive_num, require_grad=True)
            utterances = self.Tensor(game.batch_size, game.num_agents,
                                     self.vocab_size, require_grad=True)
            movements = self.Tensor(game.batch_size, game.num_entities, 
                                    self.movement_dim_size, required_grad=True).zero_()

            for agent in range(game.num_agents):
                utterance_feat = self.get_utterance_feat(game, agent)
                physical_feat = self.get_physical_feat(game, agent)
                # Divide the utterances, movement to two divisions or not?
                self.swarm_get_action(game, agent, utterance_feat, votes)
                self.scouts_get_action(game, agent, physical_feat, utterance_feat, 
                                       movements, utterances, votes)


            cost = game(votes, movements, utterances)
            self.total_cost += cost

            if self.penalizing_words:
                self.word_counter(utterances)
                utters.append(utterances)
                _, ids = utterances.view(-1, self.vocab_size).max(1)
                utters_nums_t.append(len(torch.unique(ids.view(-1))))

            if not self.training:
                timesteps.append({
                    'locations': game.locations,
                    'movements': movements,
                    'loss': cost
                })
                if self.using_utterances:
                    timesteps[-1]['utterances'] = utterances

        utters = torch.cat(utters, 0)
        if self.using_cuda:
            utters = utters.cuda()
        prob = self.word_counter.word_counts / (
            self.word_counter.oov_prob + self.word_counter.word_counts.sum() -
            1)

        # Compute reward using sum of prob based on utterances
        _, indices = utters.max(2)
        voc_cost = -torch.log(prob[indices.view(-1)]).sum()
        self.total_cost += voc_cost

        num_utters = len(torch.unique(indices))
        return self.total_cost, timesteps, num_utters, utters_nums_t, prob