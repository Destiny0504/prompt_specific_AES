import pandas as pd
import pickle
import torch
with open('/Users/chenruoting/Desktop/prompt_specific_AES/cross_prompt_attributes/1/dev.pk', 'rb') as f:
    a = pickle.load(f)
    print(type(a))
    for i in range(2, 9):
        print(len([x['prompt_id'] for x in a if x['prompt_id'] == f'{i}']))
        with open(f'/Users/chenruoting/Desktop/prompt_specific_AES/data/{i}_test.pk', 'wb') as f_write:
            # print([x for x in a if x['prompt_id'] == f'{i}'])
            pickle.dump([x for x in a if x['prompt_id'] == f'{i}'], f_write)
# EPOCH = 30
# for epoch in range(1, EPOCH + 1):
#     tau = (1 / (1 + torch.exp(torch.FloatTensor([0.99999 * (EPOCH / 2 - epoch)]))))
#     print(tau)