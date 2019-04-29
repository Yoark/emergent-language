from modules.action import ActionModule
"control swarm action: utters -> votes"

class ScoutActModule(ActionModule):
    def __init__(self, config):
        super().__init__(self, config)
        # define vote_generator structure
        self.vote_generator = nn.Sequential(

        )
    def forward(self, physical, utterance, mem, training):
        x = torch.cat((physical.squeeze(1), utterance.squeeze(1)), 1)
        processed, mem = self.processor(x, mem)
        # categorical vote
        movement = self.movement_chooser(processed)
        if training:
            utter = self.utterance_chooser(processed)
            utterance = self.gumbel_softmax(utter)
        else:
            utterance = torch.zeros(utter.size())
            if self.using_cuda:
                utterance = utterance.cuda()
            max_utter = utter.max(1)[1]
            max_utter = max_utter.data[0]
            utterance[0, max_utter] = 1

        final_movement = (movement * 2 * self.movement_step_size) - self.movement_step_size
        vote = self.vote_generator(processed)

        return vote, final_movement, utterance, mem