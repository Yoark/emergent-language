import torch
from torch import nn
from modules.gumbel_softmax import GumbelSoftmax
from modules.processing import ProcessingModule
"""control swarm action: utters -> votes"""

class ScoutActModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        # define vote_generator structure
        self.vote_generator = nn.Sequential(
             nn.Linear(config.action_processor.hidden_size, config.hidden_size),
                    nn.ELU(),
                    nn.Linear(config.hidden_size, config.num_hives)
        )
        self.gumbel_softmax_vote = GumbelSoftmax(config.use_cuda)
        
        self.movement_chooser = nn.Sequential(
        nn.Linear(config.action_processor.hidden_size, config.action_processor.hidden_size),
        nn.ELU(),
        nn.Linear(config.action_processor.hidden_size, config.movement_dim_size),
        nn.Tanh())

        self.utterance_chooser = nn.Sequential(
                    nn.Linear(config.action_processor.hidden_size, config.hidden_size),
                    nn.ELU(),
                    nn.Linear(config.hidden_size, config.vocab_size))
        self.gumbel_softmax_utter = GumbelSoftmax(config.use_cuda)
        self.processor = ProcessingModule(config.action_processor)

    def forward(self, physical, utterance, mem, training):
        x = torch.cat((physical.squeeze(1), utterance.squeeze(1)), 1)
        processed, mem = self.processor(x, mem)
        # categorical vote
        movement = self.movement_chooser(processed)
        if training:
            utter = self.utterance_chooser(processed)
            utterance = self.gumbel_softmax_utter(utter)
        else:
            utterance = torch.zeros(utter.size())
            if self.using_cuda:
                utterance = utterance.cuda()
            max_utter = utter.max(1)[1]
            max_utter = max_utter.data[0]
            utterance[0, max_utter] = 1

        final_movement = (movement * 2 * self.movement_step_size) - self.movement_step_size
        vote = self.vote_generator(processed)
        vote = self.gumbel_softmax_vote(vote)

        return vote, final_movement, utterance, mem