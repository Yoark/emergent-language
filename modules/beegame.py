import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from modules.game import GameModule
# We need physical obsearvation, utterance vector and goal vector
# And the action will out utterance and movement action
"""
    BeegameModule
    Description:
    A population of swarm consists of two groups of bees:
        -scout bees 20%
        -accumulators 80%
    !The goal for scout is to get bees to have a goal of go to a location predefined on the scouts
    !The goal for accumulators is to obtain the best possible location from scouts
    the reward is computed jointly for scouts and accumulators:
    reward = alpha * num of swarm reach consensus + beta * the quality of location of
    the majority selected hive.
    scouts:
        -movements
        -utterances
        -predefined hive info
    accumulators:
        -movements
        -utterances
    TODO



    Game consists of :
        -num_agents (scalar)
        -num_landmarks (scalar)
        -locations: [num_agents + num_landmarks, 2]
        -physical: [num_agents + num_landmarks, entity_embed_size]
        -utterances: [num_agents, vocab_size]
        -goals: [num_agents, goal_size]
        -location_observations: [num_agents, num_agents + num_landmarks, 2]
        -memories
            -utterance: [num_agents, num_agents, memory_size]
            -physical:[num_agents, num_agents + num_landmarks, memory_size]
            -action: [num_agents, memory_size]

        config needs: -batch_size, -using_utterances, -world_dim, -vocab_size, -memory_size, -num_colors -num_shapes
"""


class BeeGameModule(nn.Module):
    def __init__(self, config, num_swarm, num_scouts, num_hives):
        super().__init__()

        self.batch_size = config.batch_size
        self.using_cuda = config.use_cuda
        self.num_swarm = num_swarm
        self.num_scouts = num_scouts
        self.num_hives = num_hives
        self.num_entities = self.num_swarm + self.num_scouts + self.num_hives
        self.num_agents = self.num_swarm + self.num_scouts
        if self.using_cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor

        #! where to fit hive num in and where to fit hive locations in
        #!hives = (torch.rand(self.batch_size, self.num_agents, 1)*config.num_hives).floor()
        #!hive_value = torch.rand(self.batch_size, self.num_hives, 1)

        hive_values = torch.rand(self.batch_size, self.num_hives, 1)
        locations = torch.rand(self.batch_size, self.num_entities,
                               2) * config.world_dim
        colors = (torch.rand(self.batch_size, self.num_entities, 1) *
                  config.num_colors).floor()
        shapes = (torch.rand(self.batch_size, self.num_entities, 1) *
                  config.num_shapes).floor()

        votes = torch.zeros(self.batch_size, self.num_agents, self.num_hives)

        if self.using_cuda:
            locations = locations.cuda()
            colors = colors.cuda()
            shapes = shapes.cuda()
            votes = votes.cuda()
            hive_values = hive_values.cuda()

        self.hive_values = Variable(hive_values)
        self.votes = Variable(votes)
        # [batch_size, num_entities, 2]
        self.locations = Variable(locations)
        # [batch_size, num_entities, 2]
        self.physical = Variable(torch.cat((colors, shapes), 2).float())
        # [batch_size, num_agents, 3]

        physical_memories = torch.zeros(self.batch_size, self.num_agents,
                                        self.num_entities, config.memory_size)

        action_memories = torch.zeros(self.batch_size, self.num_agents,
                                      config.memory_size)
        vote_memories = torch.zeros(self.batch_size, self.num_agents,
                                    config.memory_size)
        if self.using_cuda:
            physical_memories = physical_memories.cuda()
            action_memories = action_memories.cuda()
            vote_memories = vote_memories.cuda()

        self.memories = {
            "vote": Variable(vote_memories),
            "physical": Variable(physical_memories),
            "action": Variable(action_memories)
        }

        utterances = torch.zeros(self.batch_size, self.num_agents,
                                 config.vocab_size)
        utterances_memories = torch.zeros(self.batch_size, self.num_agents,
                                          self.num_agents, config.memory_size)
        if self.using_cuda:
            utterances = utterances.cuda()
            utterances_memories = utterances_memories.cuda()
        self.utterances = Variable(utterances)
        self.memories["utterance"] = Variable(utterances_memories)

        #? Compute current observations of relative coordinates got from agents
        #? batch, agent, other_agent, 2
        agent_baselines = self.locations[:, :self.num_agents, :]
        #? just copy, it looks redundunct to me
        self.observations = self.locations.unsqueeze(
            1) - agent_baselines.unsqueeze(2)

    def forward(self, movements, utterances, votes):
        #? remember only scouts can move
        #? update location and compute cost
        self.locations = self.locations + movements
        agent_baselines = self.locations[:, :self.num_agents]
        self.observations = self.locations.unsqueeze(
            1) - agent_baselines.unsqueeze(2)

        self.utterances = utterances
        self.votes = votes
        return self.compute_cost(movements, votes)

    def compute_cost(self, movements, votes):
        movement_cost = self.compute_movement_cost(movements)
        vote_cost, max_freq = self.compute_vote_cost(votes)
        return vote_cost + movement_cost, max_freq

    def compute_movement_cost(self, movements):
        return torch.sum(torch.sqrt(torch.sum(torch.pow(movements, 2), -1)))

    def compute_vote_cost(self, votes):
        """ !votes: [batch, num_agents, 1]
        #! d = 100, k = 30, t = 0.7
        #! discount = d(1-sigmoid(k(max_freq(v^t) - t)))
        #! value(self, ) the quality of a hive
        #! r(s^t, a^t) = r(v^t+1) = sum(value(v_i^t+1))
        """
        #* move this into config
        d, k, t = 100, 30, 0.7

        max_freq = self.max_freq(votes)
        value = self.value(votes)
        discount = d * (1 - torch.sigmoid(k * (max_freq - t)))
        return -(value / discount).sum(), max_freq

    def value(self, votes):
        _, agent_vote = votes.max(2)
        values = self.Tensor(self.batch_size)
        for idx, hive_values in enumerate(self.hive_values):
            per_batch_value = hive_values[agent_vote[idx]].sum()
            values[idx] = per_batch_value
        return values

    def max_freq(self, votes):
        #! assume votes: [batch, num_agents, num_hives]
        #? [batch, 1]
        _, agent_vote = votes.max(2)
        per_batch_vote_count = torch.stack([(agent_vote == vote).sum(1)
                                          for vote in agent_vote.unique()]).transpose(0, 1)
        return torch.max(per_batch_vote_count.float() / self.num_agents, 1)[0]
