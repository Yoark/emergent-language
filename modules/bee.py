import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.processing import ProcessingModule
from modules.scout_action import ScoutActModule
from modules.swarm_action import SwarmActModule
from modules.word_counting import WordCountingModule
""" Swarm action defining"""


class BeeModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.init_from_config(config)
        self.total_cost = Variable(self.Tensor(1).zero_())
        # config need to be added, need to create scoutActmodule
        self.swarm_action_processor = SwarmActModule(
            config.swarm_action_processor)
        self.num_hives = config.num_hives
        self.utterance_processor = ProcessingModule(config.utterance_processor)
        self.scout_action_processor = ScoutActModule(
            config.scout_action_processor)
        self.num_scout = config.num_scout
        self.num_swarm = config.num_swarm
        self.physical_processor = ProcessingModule(config.physical_processor)
        self.physical_pooling = nn.AdaptiveAvgPool2d((1, config.feat_vec_size))
        self.utterance_pooling = nn.AdaptiveAvgPool2d((1,
                                                       config.feat_vec_size))

        self.vote_processor = ProcessingModule(config.vote_processor)
        self.vote_pooling = nn.AdaptiveAvgPool2d((1, config.feat_vec_size))

        if self.penalizing_words:
            self.word_counter = WordCountingModule(config.word_counter)

    def init_from_config(self, config):
        self.training = True
        self.using_utterances = config.use_utterances
        self.penalizing_words = config.penalize_words
        self.using_cuda = config.use_cuda
        self.time_horizon = config.time_horizon
        self.movement_dim_size = config.movement_dim_size
        self.vocab_size = config.vocab_size
        self.processing_hidden_size = config.physical_processor.hidden_size
        self.Tensor = torch.cuda.FloatTensor if self.using_cuda else torch.FloatTensor

    def reset(self):
        self.total_cost = torch.zeros_like(self.total_cost)
        if self.using_utterances and self.penalizing_words:
            if self.using_cuda:
                self.word_counter.word_counts = torch.zeros(
                    self.vocab_size).cuda()
            else:
                self.word_counter.word_counts = torch.zeros(self.vocab_size)

    def update_mem(self, game, mem_str, new_mem, agent, other_agent=None):
        # TODO: Look into tensor copying from Variable
        new_big_mem = Variable(self.Tensor(game.memories[mem_str].data))
        if other_agent is not None:
            new_big_mem[:, agent, other_agent] = new_mem
        else:
            new_big_mem[:, agent] = new_mem
        game.memories[mem_str] = new_big_mem

    def swarm_get_action(self, game, agent, utterance_feat, votes, utterances):
        #* Good
        """
        """
        #! WHat memory should it use?
        utterance, vote, new_mem = self.swarm_action_processor(
            utterance_feat, game.memories["action"][:, agent], self.training)
        self.update_mem(game, "action", new_mem, agent)
        votes[:, agent, :] = vote
        utterances[:, agent, :] = utterance

    def scouts_get_action(self, game, agent, physical_feat, utterance_feat,
                          movements, utterances, votes):
        #* Good
        vote, movement, utterance, new_mem = self.scout_action_processor(
            physical_feat, utterance_feat, game.memories["action"][:, agent],
            self.training)
        self.update_mem(game, "action", new_mem, agent)
        movements[:, agent, :] = movement
        votes[:, agent, :] = vote
        utterances[:, agent, :] = utterance

    def process_utterances(self, game, agent, other_agent,
                           utterance_processes):
        #* Good
        utterance_processed, new_mem = self.utterance_processor(
            game.utterances[:, other_agent],
            game.memories["utterance"][:, agent, other_agent])
        self.update_mem(game, "utterance", new_mem, agent, other_agent)
        utterance_processes[:, other_agent, :] = utterance_processed

    def process_physical(self, game, agent, other_entity, physical_processes):
        #* Good
        physical_processed, new_mem = self.physical_processor(
            torch.cat((game.observations[:, agent, other_entity],
                       game.physical[:, other_entity]), 1),
            game.memories["physical"][:, agent, other_entity])
        self.update_mem(game, "physical", new_mem, agent, other_entity)
        physical_processes[:, other_entity, :] = physical_processed

    # def process_vote(self, game, agent, other_agent, vote_processes):
    #     vote_processed, new_mem = self.vote_processor(
    #         game.votes[:, other_agent], game.memories["vote"][:, agent, other_agent])
    #     self.update_mem(game, "votes", new_mem, agent, other_agent)
    #     vote_processes[:, other_agent, :] = vote_processed

    def get_utterance_feat(self, game, agent):
        #* Good
        """
        gets observed utterances feature vector for a agent.
        """
        utterance_processes = Variable(
            self.Tensor(game.batch_size, game.num_scouts,
                        self.processing_hidden_size))
        for other_agent in range(game.num_scouts):
            self.process_utterances(game, agent, other_agent,
                                    utterance_processes)
        #! [1, feat_vec_size]
        return self.utterance_pooling(utterance_processes)

    def get_physical_feat(self, game, agent):
        physical_processes = Variable(
            self.Tensor(game.batch_size, game.num_entities,
                        self.processing_hidden_size))
        for entity in range(game.num_entities):
            self.process_physical(game, agent, entity, physical_processes)
        return self.physical_pooling(physical_processes)

    # def get_vote_feat(self, game, agent):
    #     vote_processes = Variable(self.Tensor(game.batch_size, game.num_agents,
    #                         self.processing_hidden_size))
    #     for voter in range(game.num_agents):
    #         self.process_vote(game, agent, voter, vote_processes)
    #     return self.vote_pooling(vote_processes)

    def forward(self, game):
        #* Good
        timesteps = []
        utters = []
        utters_nums_t = []
        votes_epoch = []
        votes_ratio_t = []
        for t in range(self.time_horizon):

            votes = Variable(
                self.Tensor(game.batch_size, game.num_agents, self.num_hives))
            utterances = Variable(
                self.Tensor(game.batch_size, game.num_agents, self.vocab_size))
            movements = Variable(
                self.Tensor(game.batch_size, game.num_entities,
                            self.movement_dim_size)).zero_()

            for agent in range(game.num_swarm):
                utterance_feat = self.get_utterance_feat(game, agent)
                #? vote_feat = self.get_vote_feat(game, agent)
                # ? physical_feat = self.get_physical_feat(game, agent)
                # Divide the utterances, movement to two divisions or not?
                self.swarm_get_action(game, agent, utterance_feat, votes,
                                      utterances)

            for agent in range(game.num_swarm,
                               game.num_swarm + game.num_scouts):
                utterance_feat = self.get_utterance_feat(game, agent)
                # vote_feat = self.get_vote_feat(game, agent)
                physical_feat = self.get_physical_feat(game, agent)
                self.scouts_get_action(game, agent, physical_feat,
                                       utterance_feat, movements, utterances,
                                       votes)
            votes_epoch.append(votes)
            cost, max_freq = game(movements, utterances, votes)
            votes_ratio_t.append(max_freq)
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
        votes_epoch = torch.cat(votes_epoch, 0)
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
        return self.total_cost, timesteps, num_utters, utters_nums_t, prob, votes_epoch, votes_ratio_t
