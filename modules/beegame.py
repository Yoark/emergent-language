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

    def __init__(self, config, num_agents, num_landmarks):
        super().__init__(config, num_agents, num_landmarks)

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

