from .d2prune_utils import D2SparseGPT, D2Wanda
from  .prune_opt import D2Prune_OPT
from .prune_llama import D2Prune_LLaMA

class D2Prune:
    def __init__(self, args, model):
        self.args = args
        if 'llama' in self.args.model.lower():
            self.pruner = D2Prune_LLaMA(args, model)
        elif 'opt' in self.args.model.lower():
            self.pruner = D2Prune_OPT(args, model)
        else:
            raise ValueError(f'Unsupported model {self.args.model.lower()}, please check your model path')