import torch
import torch.nn as nn
from torch.autograd import Variable


class WordCountingModule(nn.Module):
    def __init__(self, config):
        super(WordCountingModule, self).__init__()
        self.oov_prob = config.oov_prob
        self.use_cuda = config.use_cuda
        word_counts = torch.zeros(config.batch_size, config.vocab_size)
        if config.use_cuda:
            word_counts = word_counts.cuda()
        self.word_counts = Variable(word_counts)

    def forward(self, utterances, timestep):
        batch_size, num_agents, vocab_size = utterances.shape
        _, indices = utterances.max(2)
        indicator = torch.zeros(batch_size, vocab_size)
        if self.use_cuda:
            indicator = indicator.cuda()
        batch = 0
        for agent_idx in indices:
            for _index in agent_idx:
                indicator[batch, _index] += 1
            batch += 1
        # assume agents make an utterance each time step
        n = (timestep + 1) * num_agents
        alpha = self.oov_prob
        self.word_counts = self.word_counts + indicator
        prob_ck = self.word_counts / (alpha + n - 1)
        import ipdb; ipdb.set_trace()
        reward = 0
        batch = 0
        for agent_idx in indices:
            for _index in agent_idx:
                reward += torch.log(prob_ck[batch, _index])
            batch += 1
        return -reward, prob_ck
