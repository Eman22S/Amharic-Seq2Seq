import yaml
import torch
import copy
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda')
filename = 'tr_5792_tr58093.npy'
#solver = Solver(config,paras)
#solver.load_data()
#verbose('Load ASR model from ' + os.path.join(self.ckpdir))
asr_model = torch.load('asr')
x = torch.FloatTensor(np.load(filename))
x = pad_sequence(x, batch_first=True)
mfcc_len = x.shape[0]
x = np.reshape(x, (1, mfcc_len, 80))

state_len = torch.sum(torch.sum(x.cpu(), dim=-1) != 0, dim=-1)
state_len = [int(sl) for sl in state_len]
max_decode_step = int(np.ceil(state_len[0] * 0.1))
model = copy.deepcopy(asr_model).to(device)
decode_beam_size = 20
x = x.cuda()
#hyps = model.beam_decode(x, max_decode_step, state_len, decode_beam_size)

hyp = asr_model(x,max_decode_step)
