from modules.action import ActionModule
"control swarm action: utters -> votes"

class ScoutActModule(ActionModule):
    def __init__(self, config):
        super().__init__(self, config)
        # define vote_generator structure
        self.vote_generator = nn.Sequential(

        )
    def forward(self, utterance=None, mem, training):
        x = utterance.squeeze(1)
        processed, mem = self.processor(x, mem)
        # categorical vote
        vote = self.vote_generator(processed)
        return vote, movement, utterance, mem