from turtle import forward
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer):
    """Initialize a layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect pooling method. ")

        return x


class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True
                                                 )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True
                                                 )

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8,
                                               freq_stripes_num=2
                                               )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)

        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input):  # TODO: I have deleted mixup part here.
        """
        Input: (batch_size, data_length)
        """

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)  # (batch_size, freq_bins, time_steps, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)

        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        # TODO: Even though the embedding is above, the `embedding` we actually use will be this clipwise output.
        # TODO: The clipwise_output is the action vector.
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding
        }

        return output_dict


class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins,
                 fmin, fmax, classes_num, freeze_base):
        """Classifier to a new task. Our model actually begins here. """
        super(Transfer_Cnn14, self).__init__()
        # TODO: the class num here is the classnum for CNN14.
        audioset_classes_num = 527

        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,
                          fmax, audioset_classes_num)

        # Transfer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input)
        embedding = output_dict['embedding']
        action_vector = output_dict['clipwise_output']

        # TODO: concat the input with the action vector. But here for easy implementation,
        # we ignore this step.

        output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)

        outputs = {}
        outputs['embedding'] = embedding
        outputs['action_vector'] = action_vector
        outputs['clipwise_output'] = output

        return outputs


import torch.nn as nn
import torch


class ProjectionLayer(nn.Module):
    """ Classification Layer.
    """

    def __init__(self, input_size, units):
        super().__init__()
        self.input_size = input_size
        self.units = units

        # Projection
        self.projection = nn.Linear(input_size, self.units, bias=False)

        self.init_weight()

    def init_weight(self):
        init_layer(self.projection)

    def forward(self, input):
        x = input
        output = self.projection(input)
        return output

import torch.nn as nn
import torch.nn.functional as F
# from pytorch_transformers import BertTokenizer
# from pytorch_transformers import BertModel
from transformers import BertTokenizer, BertModel
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 21


class bertEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.modelInput = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    def forward(self, x):
        # result = []
        # for token in x:
        #   tokens_tensor = torch.LongTensor([token]).cuda()
        #   outputs = self.modelInput(tokens_tensor)
        #   pooled_output2 = outputs[1]  
        #   result.append(pooled_output2)
        # output = torch.cat(result,dim=0)

        
        encoding = self.tokenizer(x, add_special_tokens = True, return_tensors="pt", padding='max_length', max_length=MAX_LEN, truncation=True).to(device)
        output = self.modelInput(**encoding)
        # last_hidden_states shape (32, 768)

        return output.pooler_output

class UnFlatten(nn.Module):
    def __init__(self):
        super(UnFlatten, self).__init__()
    def forward(self, inp):
        return inp.view(inp.size(0), 1)

class ConcatCLS(nn.Module):
    """ Classification Layer.
    """
    def __init__(self, sample_rate, window_size, hop_size, mel_bins,
                 fmin, fmax, classes_num, freeze_base, text_input_size = 768, audio_input_size = 2048, units = 1024):
        super().__init__()
        self.bert_encoder = bertEmbedding()  # bertEmbedding()
        audioset_classes_num = 527
        # self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num
        self.audio_encoder = Cnn14(sample_rate, window_size, hop_size, mel_bins,
                                   fmin, fmax, audioset_classes_num)  # Cnn14()
        self.project_bert = ProjectionLayer(text_input_size, units)
        self.project_audio = ProjectionLayer(audio_input_size, units)

        self.last = nn.Linear(units,1, bias=False)
        self.sig = nn.Sigmoid()

        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.unfla = UnFlatten()
        # self.sig = nn.Sigmoid()

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.audio_encoder.load_state_dict(checkpoint['model'])

    def forward(self, audio_input, text_input):
        text_output = self.bert_encoder(text_input)  # shape: (bs, hidden)
        audio_output = self.audio_encoder(audio_input)  # shape: (bs, 2048)
        p_bert = self.project_bert(text_output)  # output shape: (batch_size, unit)
        p_audio = self.project_audio(audio_output['embedding'])  # shape: (batch_size, unit)

        # product
        logits = torch.mul(p_bert, p_audio)
        logits = self.last(logits)
        logits = self.sig(logits)
        return logits

        # cosine similarity
        # cos_sim = self.cos(p_bert, p_audio)
        # unflatten_sim = self.unfla(cos_sim)
        # sig_cos = self.sig(unflatten_sim)
        # return sig_cos




