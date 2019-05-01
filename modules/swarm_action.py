import torch
import torch.nn as nn

from modules.processing import ProcessingModule
from modules.gumbel_softmax import GumbelSoftmax

from modules.action import ActionModule
"control swarm action: utters -> votes"


"""BeeActModule, takes physical feature vector, utterance feature vector, and vote feature vector (only for scouts)
concatenation of three feature goes into a processing module, vote feature vector goes to a processing module, their
outputs and fed into two indepedent network to output utterance, movement, vote for scouts, vote only for swarm.

Returns:
    [type] -- [description]
"""
class SwarmActModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        # define vote_generator structure
        self.vote_generator = nn.Sequential(
            nn.Linear(config.action_processor.hidden_size, config.hidden_size),
            nn.ELU(),
            nn.Linear(config.hidden_size, config.num_hives)
        )
        self.gumbel_softmax = GumbelSoftmax(config.use_cuda)

        self.utterance_chooser = nn.Sequential(
                    nn.Linear(config.action_processor.hidden_size, config.hidden_size),
                    nn.ELU(),
                    nn.Linear(config.hidden_size, config.vocab_size))
        self.gumbel_softmax_utter = GumbelSoftmax(config.use_cuda)
        self.processor = ProcessingModule(config.action_processor)

    def forward(self, utterance, mem, training):
        x = utterance.squeeze(1)
        processed, mem = self.processor(x, mem)
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
        vote = self.vote_generator(processed)
        gumbel_vote = self.gumbel_softmax(vote)
        return utterance, gumbel_vote, mem
