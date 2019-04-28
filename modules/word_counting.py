import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import torch


class WordCountingModule(nn.Module):
    def __init__(self, config):
        super(WordCountingModule, self).__init__()
        self.oov_prob = config.oov_prob
        self.word_counts = torch.zeros(config.vocab_size)
        self.use_cuda = config.use_cuda
        if config.use_cuda:
            self.word_counts = self.word_counts.cuda()

    def forward(self, utterances):
        batch_size, num_agents, vocab_size = utterances.shape
        _, indices = utterances.max(2)
        # over all batches
        indicator = torch.zeros(vocab_size)
        if self.use_cuda:
            indicator = indicator.cuda()

        for _index in indices.view(-1):
            indicator[_index] += 1

        # assume agents make an utterance each time step
        self.word_counts = self.word_counts + indicator
