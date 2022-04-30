import torch
import numpy as np
from pytorch_transformers import BertTokenizer
from pytorch_transformers import BertModel

## Load pretrained model/tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x
    
    return x.to(device)

def append_to_dict(dct, key, val):
    if key in dct.keys():
        dct[key].append(val)
    else:
        dct[key] = [val]

def forward(model, generator, return_input=False, return_target=False):
    output_dict = {}
    device = next(model.parameters()).device
    i = 0
    for batch_data_dict in generator:
        print(i)
        i += 1
        batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)

        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform)
            #batch_output = model(batch_data_dict['waveform'], [tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[CLS] " + element.decode("utf-8") + " [SEP]")) for element in batch_data_dict['caption']])
        
        append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])

        append_to_dict(output_dict, 'clipwise_output',
            batch_output['clipwise_output'].data.cpu().numpy()
        )

        if return_input:
            append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])
        
        if return_target:
            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target'])
    
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
        
    return output_dict
