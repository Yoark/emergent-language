import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import torch

class WordCountingModule(nn.Module):
    def __init__(self, config):
        super(WordCountingModule, self).__init__()
        self.oov_prob = config.oov_prob
        self.word_counts = torch.zeros(config.vocab_size)
        if config.use_cuda:
            self.word_counts = self.word_counts.cuda()

    def forward(self, utterances):
        #cost = -(utterances/(self.oov_prob + self.word_counts.sum() - 1)).sum()
        self.word_counts = self.word_counts + utterances

        #return cost
