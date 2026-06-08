from __future__ import print_function

from networks.GATE import GATE
from networks.GATE_multi import GATE_multi

def prepare_model(opt):
    model = eval(opt.model)(opt)
    return model