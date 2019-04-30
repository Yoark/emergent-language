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
            self.Tensor = torch.CudaFloatTensorBase
        locations = torch.rand(self.batch_size, self.num_entities,
                               2) * config.world_dim

        #! where to fit hive num in and where to fit hive locations in
        #!hives = (torch.rand(self.batch_size, self.num_agents, 1)*config.num_hives).floor()
        #!hive_value = torch.rand(self.batch_size, self.num_hives, 1)

        votes = torch.zeros(self.batch_size, self.num_agents,config.num_swarm)
        if self.using_cuda:
            votes = votes.cuda()
        self.votes = Variable(votes)

        #? the goal hive location label
        """ goal_entities = (torch.rand(self.batch_size, self.num_agents, 1)*
                        self.num_hives).floor().long() """
        """ goal_locations = self.Tensor(self.batch_size, self.num_agents, 2)
        #? the agent label
        goal_agents = self.Tensor(self.batch_size, self.num_agents, 1) """

        if self.using_cuda:
            locations = locations.cuda()
            # hives = hives.cuda()
            # goal_entities = goal_entities.cuda()

        self.locations = Variable(locations)
        # self.hives = hives.float()

        
"""         for b in range(self.batch_size):
            goal_agents[b] = torch.randperm(self.num_agents).view(self.num_agents, -1)

        for b in range(self.batch_size):
            goal_locations[b] = self.locations[b][goal_entities[b].squeeze()] """

        #? cat goal labels with its initial goal coordinates.
"""         self.goals = torch.cat((goal_locations, goal_agents), 2).requires_grad()
        goal_agents.requires_grad_() """
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
                "vote":
                Variable(
                    torch.zeros(self.batch_size, self.num_agents,
                                self.num_agents, config.memory_size).cuda())
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
                "vote":
                Variable(
                    torch.zeros(self.batch_size, self.num_agents,
                                self.num_agents, config.memory_size)
            }
        if self.using_utterances:
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
        agent_baselines = self.locations[:, :self.num_agent, :]
        #? just copy, it looks redundunct to me
        self.observations = self.locations.unsqueeze(1) - agent_baselines.unsqueeze(2)

        #new_obs = self.goals[:,:,:2] - agent_baselines
        #self.observed_goals = torch.cat((new_obs, goal_agents), dim=2)



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
        def forward(self, movements, utterances, votes):
            #? remember only scouts can move
            #? update location and compute cost
            self.locations = self.locations + movements
            agent_baselines = self.locations[:, :self.num_agents]
            self.observations = self.locations.unsqueeze(
            1) - agent_baselines.unsqueeze(2)
            #new_obs = self.goals[:, :, :2] - agent_baselines
            #goal_agents = self.goals[:, :, 2].unsqueeze(2)
            # self.observed_goals = torch.cat((new_obs, goal_agents), 2)

            self.utterances = utterances
            self.votes = votes
            return self.compute_cost(movements, utterances, votes)

        def compute_cost(self, movements, utterances, votes):

            movement_cost = self.compute_movement_cost(movements)
            vote_cost = self.compute_vote_cost(votes)

            return physical_cost + goal_pred_cost + movement_cost

        def compute_movement_cost(self, mvoements):
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

            discount = d * (1 - torch.sigmoid(k*max_freq(votes) - t))
            return -torch.sum(torch.sum(value(), 1)/discount)

        def value(self):
            #? value = 1/square(distance to center of swarm times a constant)
            swarm_center = torch.mean(self.locations[:, self.num_swarm, :], [1])
            #! (batch, num_agents, 1)
            return torch.sum(torch.pow(1/(self.goal_locations - swarm_center.unsqueeze(1)), 2), 2)

        def max_freq(self, votes):
            #! assume votes: [batch, num_agents, num_hives]
            #? [batch, 1]
            return torch.max(torch.sum(votes, 1), 1)/self.num_agents










