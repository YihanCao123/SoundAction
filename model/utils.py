import torch
import numpy as np


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
    for batch_data_dict in generator:
        batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)

        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform, [element for element in batch_data_dict['caption']])
        
        append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])

        append_to_dict(output_dict, 'predict_target', batch_output.data.cpu().numpy())

        if return_input:
            append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])
        
        if return_target:
            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target'])

                
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
        
    return output_dict
