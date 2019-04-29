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

class BeeGameModule(GameModule):

    def __init__(self, config, num_swarms, num_scouts, num_hive):
        super().__init__()

        self.batch_size = config.batch_size
        self.using_cuda = config.use_cuda
        self.num_swarms = num_swarms
        self.num_scouts = num_scouts
        self.num_hive = num_hive
        self.num_entities = self.num_swarms + self.num_scouts + self.num_hive
        self.num_agents = self.num_swarms + self.num_scouts
        if self.using_cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.CudaFloatTensorBase
        
        locations = torch.rand(self.batch_size, self.num_entities,
                               2) * config.world_dim
        hive_values = torch.rand(self.batch_size, self.num_entities, 1)

        
        #? the physical feature
        goal_entities = (torch.rand(self.batch_size, self.num_agents, 1)*
                        self.num_hive).floor().long()
        goal_locations = self.Tensor(self.batch_size, self.num_agents, 2)

        goal_agents = self.Tensor(self.batch_size, self.num_agents, 1)
        if using_cuda:
            locations = locations.cuda()

        self.locations = locations.require_grad_()
        self.physical = hive_values.float()

        #? set goals for swarms and scouts
        for b in range(self.batch_size):
            goal_agents[b] = torch.randperm(self.num_agents).view(self.num_agents, -1)
        
        for b in range(self.batch_size):
            goal_locations[b] = self.locations[b][goal_entities[b].squeeze()]
        
        self.goals = torch.cat((goal_locations, goal_agents), 2).requires_grad()
        goal_agents.requires_grad_()
        #? mem unchanged
        if self.using_cuda:
            self.memories = {
                "physical":
                Variable(
                    torch.zeros(self.batch_size, self.num_agents,
                                self.num_entities, config.memory_size).cuda()),
                "action":
                Variable(
                    torch.zeros(self.batch_size, self.num_agents,
                                config.memory_size).cuda())

            }
        else:
            self.memories = {
                "physical":
                Variable(
                    torch.zeros(self.batch_size, self.num_agents,
                                self.num_entities, config.memory_size)),
                "action":
                Variable(
                    torch.zeros(self.batch_size, self.num_agents,
                                config.memory_size))
            }
        if self.using_cuda:
            self.utterances = Variable(
                    torch.zeros(self.batch_size, self.num_agents,
                                config.vocab_size).cuda())
                self.memories["utterance"] = Variable(
                    torch.zeros(self.batch_size, self.num_agents,
                                self.num_agents, config.memory_size).cuda())
        
        agent_baselines = self.locations[:, :self.num_agent, :]

        


"""
- The scouts should be able to advertise to its surrounding bees within a radius
- Bees can observe its surroundings within a radius
!utterance remains unchanged within one epsiode of game.
Each timestamp, scouts location updates, goal updates, accumulators location updates,
goal updates, utterance might get updated,
! and a accrued cost for all games in the batch is computed
! More cost computation methods:
    -Compute the consensus loss of bee as negative sum of log 2norm distance between acquired goals of each pair of bees
    -Compute the qulity loss as negative sum of log distance between best possible goal and each bees's goal.
"""


