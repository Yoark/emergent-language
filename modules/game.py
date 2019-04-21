import torch
import torch.nn as nn
from torch.autograd import Variable


class GameModule(nn.Module):
    """
    The GameModule takes in all actions(movement, utterance, goal prediction)
    of all agents for a given timestep and returns the total cost for that
    timestep.

    Game consists of:
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

    def __init__(self, config, num_agents, num_landmarks):
        super(GameModule, self).__init__()

        self.batch_size = config.batch_size  # scalar: num games in this batch
        self.using_utterances = config.use_utterances  # bool: whether current batch allows utterances
        self.using_cuda = config.use_cuda
        self.num_agents = num_agents  # scalar: number of agents in this batch
        self.num_landmarks = num_landmarks  # scalar: number of landmarks in this batch
        self.num_entities = self.num_agents + self.num_landmarks  # type: int

        if self.using_cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor

        locations = torch.rand(self.batch_size, self.num_entities,
                               2) * config.world_dim
        colors = (torch.rand(self.batch_size, self.num_entities, 1) *
                  config.num_colors).floor()
        shapes = (torch.rand(self.batch_size, self.num_entities, 1) *
                  config.num_shapes).floor()

        goal_agents = self.Tensor(self.batch_size, self.num_agents, 1)
        goal_entities = (torch.rand(self.batch_size, self.num_agents, 1) *
                         self.num_landmarks).floor().long() + self.num_agents
        goal_locations = self.Tensor(self.batch_size, self.num_agents, 2)

        if self.using_cuda:
            locations = locations.cuda()
            colors = colors.cuda()
            shapes = shapes.cuda()
            goal_entities = goal_entities.cuda()

        # [batch_size, num_entities, 2]
        self.locations = Variable(locations)
        # [batch_size, num_entities, 2]
        self.physical = Variable(torch.cat((colors, shapes), 2).float())

        #TODO: Bad for loop?
        for b in range(self.batch_size):
            goal_agents[b] = torch.randperm(self.num_agents).view(
                self.num_agents, -1)

        for b in range(self.batch_size):
            goal_locations[b] = self.locations.data[b][
                goal_entities[b].squeeze()]

        # [batch_size, num_agents, 3]
        self.goals = Variable(torch.cat((goal_locations, goal_agents), 2))
        goal_agents = Variable(goal_agents)

        mem_physical = torch.zeros(self.batch_size, self.num_agents,
                                   self.num_entities, config.memory_size)
        mem_action = torch.zeros(self.batch_size, self.num_agents,
                                 config.memory_size)
        if self.using_cuda:
            mem_physical = mem_physical.cuda()
            mem_action = mem_action.cuda()

        self.memories = {
            "physical": Variable(mem_physical),
            "action": Variable(mem_action)
        }

        if self.using_utterances:
            utterances = torch.zeros(self.batch_size, self.num_agents,
                                     config.vocab_size)
            remembered_utterances = torch.zeros(
                self.batch_size, self.num_agents, self.num_agents,
                config.memory_size)
            if self.using_cuda:
                utterances = utterances.cuda()
                remembered_utterances = remembered_utterances.cuda()

            self.utterances = Variable(utterances)
            self.memories["utterance"] = Variable(remembered_utterances)

        agent_baselines = self.locations[:, :self.num_agents, :]

        sort_idxs = torch.sort(self.goals[:, :, 2])[1]
        self.sorted_goals = Variable(self.Tensor(self.goals.size()))
        # TODO: Bad for loop?
        for b in range(self.batch_size):
            self.sorted_goals[b] = self.goals[b][sort_idxs[b]]
        self.sorted_goals = self.sorted_goals[:, :, :2]

        # [batch_size, num_agents, num_entities, 2]
        self.observations = self.locations.unsqueeze(
            1) - agent_baselines.unsqueeze(2)

        new_obs = self.goals[:, :, :2] - agent_baselines

        # [batch_size, num_agents, 2] [batch_size, num_agents, 1]
        self.observed_goals = torch.cat((new_obs, goal_agents), dim=2)

    def forward(self, movements, goal_predictions, utterances):
        """
        Updates game state given all movements and utterances and returns accrued cost
            - movements: [batch_size, num_agents, config.movement_size]
            - utterances: [batch_size, num_agents, config.utterance_size]
            - goal_predictions: [batch_size, num_agents, num_agents, config.goal_size]
        Returns:
            - scalar: total cost of all games in the batch
        """
        self.locations = self.locations + movements
        agent_baselines = self.locations[:, :self.num_agents]
        self.observations = self.locations.unsqueeze(
            1) - agent_baselines.unsqueeze(2)
        new_obs = self.goals[:, :, :2] - agent_baselines
        goal_agents = self.goals[:, :, 2].unsqueeze(2)
        print('self.goals.shape: {}'.format(self.goals.shape))
        print('new_obs.shape: {}'.format(new_obs.shape))
        print('goal_agents.shape: {}'.format(goal_agents.shape))

        self.observed_goals = torch.cat((new_obs, goal_agents), dim=2)
        print('self.observed_goals.shape: {}'.format(
            self.observed_goals.shape))
        if self.using_utterances:
            self.utterances = utterances
            return self.compute_cost(movements, goal_predictions, utterances)
        else:
            return self.compute_cost(movements, goal_predictions)

    def compute_cost(self, movements, goal_predictions, utterances=None):
        physical_cost = self.compute_physical_cost()
        movement_cost = self.compute_movement_cost(movements)
        goal_pred_cost = self.compute_goal_pred_cost(goal_predictions)
        return physical_cost + goal_pred_cost + movement_cost

    def compute_physical_cost(self):
        """
        Computes the total cost agents get from being near their goals
        agent locations are stored as [batch_size, num_agents + num_landmarks, entity_embed_size]
        """
        return 2 * self.get_avg_agent_to_goal_distance()

    def compute_goal_pred_cost(self, goal_predictions):
        """
        Computes the total cost agents get from predicting others' goals
        goal_predictions: [batch_size, num_agents, num_agents, goal_size]
        goal_predictions[., a_i, a_j, :] = a_i's prediction of a_j's goal
        with location relative to a_i

        We want:
            real_goal_locations[., a_i, a_j, :] = a_j's goal with location relative to a_i

        We have:
            goals[., a_j, :] = a_j's goal with absolute location
            observed_goals[., a_j, :] = a_j's goal with location relative to a_j

        Which means we want to build an observed_goals-like tensor but relative to each agent
            real_goal_locations[., a_i, a_j, :] = goals[., a_j, :] - locations[a_i]
        """
        relative_goal_locs = self.goals.unsqueeze(
            1)[:, :, :, :2] - self.locations.unsqueeze(
                2)[:, :self.num_agents, :, :]
        goal_agents = self.goals.unsqueeze(1)[:, :, :, 2:].expand_as(
            relative_goal_locs)[:, :, :, -1:]
        relative_goals = torch.cat((relative_goal_locs, goal_agents), dim=3)
        return torch.sum(
            torch.sqrt(
                torch.sum(torch.pow(goal_predictions - relative_goals, 2),
                          -1)))

    def compute_movement_cost(self, movements):
        """
        Computes the total cost agents get from moving
        """
        return torch.sum(torch.sqrt(torch.sum(torch.pow(movements, 2), -1)))

    def get_avg_agent_to_goal_distance(self):
        return torch.sum(
            torch.sqrt(
                torch.sum(
                    torch.pow(
                        self.locations[:, :self.num_agents, :] -
                        self.sorted_goals, 2), -1)))
