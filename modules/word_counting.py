import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable

class WordCountingModule(nn.Module):
    def __init__(self, config):
        super(WordCountingModule, self).__init__()
        self.oov_prob = config.oov_prob
        word_counts = Tensor(config.vocab_size)
        if config.use_cuda:
            word_counts.cuda()
        self.word_counts = Variable(word_counts)

    def forward(self, utterances):
        import ipdb
        cost = -(utterances/(self.oov_prob + self.word_counts.sum() - 1)).sum()
        ipdb.set_trace()

        self.word_counts = self.word_counts + utterances

        ipdb.set_trace()
        return cost
