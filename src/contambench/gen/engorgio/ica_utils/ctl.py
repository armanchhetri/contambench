# Engorgio (Dong et al., ICLR 2025, "An Engorgio Prompt Makes Large Language Model Babble on").
# Adapted by the ContamBench authors from https://github.com/jianshuod/Engorgio-prompt
# (commit 5fb50dd). The upstream repository carries no license file; see THIRD_PARTY_NOTICES.md.


class Converge_check():
    def __init__(self,converge_count:int = 50) -> None:
        self.counter = 0
        self.new_state = None
        self.converge_count = converge_count
    
    def __call__(self, new_trigger_tokens):
        if self.new_state is None: 
            self.new_state = new_trigger_tokens
            return False

        if check_the_same(self.new_state, new_trigger_tokens):
            self.counter += 1
        else:
            self.new_state = new_trigger_tokens
            self.counter = 0

        return self.counter >=self.converge_count


def check_the_same(l1, l2):
    res = True
    dis_list = []
    for idx, (l1x, l2x) in enumerate(zip(l1, l2)):
        if l1x != l2x: 
            res = False
            dis_list.append(idx)
    return res