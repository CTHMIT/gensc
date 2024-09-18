# -*- coding: utf-8 -*-
"""
Project Name: GenSC: Generative Semantic Communication Systems Using BART-Like Model
Author: Chun-Tse Hsu 
Date: 2024-08-27
Description: This script contains the implementation of the GenSC model, including model training, and evaluation.
"""

import os
import gc
import math
import pickle as pkl
import time
import json
from random import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Tuple, Any
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor, device
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util

class Config():
    snr_noise_list=[0,3,6,9,12,15,18]

    in_features=16
    d_model=128
    vocab_size=None
    max_position_embeddings=128

    num_layers=3
    ffn_dim=512
    attention_heads=8
    activation_function="gelu"
    
    dropout=0.1
    attention_dropout=0.1
    activation_dropout=0.1
    scale_embedding=False
    
    bos_token_id=None
    pad_token_id=None
    eos_token_id=None
    msk_token_id=None  


def _expand_mask(mask, dtype, tgt_len = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _make_causal_mask(input_ids_shape, dtype, past_key_values_length = 0):

    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings + 2, embedding_dim)
        self.offset = 2

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, device=self.weight.device
        ).expand(bsz, -1) + self.offset
        return super().forward(positions)

class Attention(nn.Module):
    def __init__(self,  embed_dim,  num_heads, dropout = 0.1, is_decoder=False, bias=True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


        
    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, key_value_states=None, past_key_value=None,
                attention_mask=None):

        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
        
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)


        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)


        return attn_output, past_key_value
    
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.d_model        
        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = F.gelu
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)        
        self.input_layer_norm = nn.LayerNorm(self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states, attention_mask):

        residual = hidden_states  
        hidden_states, _  = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value) if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ) else hidden_states

        outputs = (hidden_states,)

        return outputs

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = Attention(self.embed_dim, config.attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.encoder_attn = Attention(self.embed_dim, config.attention_heads, dropout=config.attention_dropout, is_decoder=True)
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)
        self.input_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim) 
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = F.gelu
        self.activation_dropout = config.activation_dropout   

    def forward(self, 
        hidden_states, 
        attention_mask=None, 
        encoder_hidden_states=None, 
        encoder_attention_mask=None, 
        past_key_value=None,
        ):
    
        residual = hidden_states       

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attn_present_key_value = None

        if encoder_hidden_states is not None:
            residual = hidden_states            

            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            present_key_value = present_key_value + cross_attn_present_key_value

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        return outputs
    
class Encoder(nn.Module):
    def __init__(self, config, embed_tokens = None):
        super(Encoder, self).__init__()
        self.dropout = config.dropout
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight
        self.learn_embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        

    def forward(self, input_ids, attention_mask,  head_mask=None, inputs_embeds=None):
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale


        learning = self.learn_embed_positions(input).to(inputs_embeds.device)      
           
        hidden_states = inputs_embeds + learning 
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                    )

            hidden_states = layer_outputs[0]


        return hidden_states
              

class Decoder(nn.Module):
    def __init__(self, config, embed_tokens=None):
        super(Decoder, self).__init__()
        self.config = config
        self.dropout = config.dropout
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight
        self.learn_embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    def forward(self, 
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values = None,
        inputs_embeds = None):

 
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            inpt = input_ids
            input_shape = inpt.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            inpt = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(inpt) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        learning = self.learn_embed_positions(inpt, past_key_values_length).to(inputs_embeds.device)

        hidden_states = inputs_embeds + learning
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for idx, decoder_layer in enumerate(self.layers):
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                )
            hidden_states = layer_outputs[0]  
       
        return hidden_states


class LoadDataset(Dataset):
    def __init__(self, args, name='train'):
        
        with open(args.data_path + '{}_data.pkl'.format(name), 'rb') as f:
            self.data = pkl.load(f)

    def __getitem__(self, index):
        sents = self.data[index]
        return  sents

    def __len__(self):
        return len(self.data)


def initNetParams(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def loss_function(logits, trg_real, padding_idx, Loss_Func):

    pred = logits.contiguous().view(-1, logits.size(-1))
    trg = trg_real.contiguous().view(-1)

    loss = Loss_Func(pred, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    loss *= mask
    
    return loss.mean()



def valid_step(model, src, fading_noise, pad, Loss_Func, channel):
    
    trg_inp = src[:, :-1] 
    trg_real = src[:, 1:]    
    
    src_padding_mask, trg_padding_mask = create_masks(src, trg_inp, pad)

    encoder_input = src

    Tx = model.encoder(
        input_ids=encoder_input,
        attention_mask=src_padding_mask,
        )   

    channel_enc_inp = model.channel_encoder(Tx)
    fading_signal = channel_Fading(channel_enc_inp, fading_noise, channel) 
    Rx = model.channel_decoder(fading_signal)  
    dec_output = model.decoder(
            input_ids=trg_inp, 
            encoder_hidden_states=Rx, 
            attention_mask=trg_padding_mask, 
            encoder_attention_mask=src_padding_mask,
            )

    logits = model.lm_head(dec_output)
    logits = logits + model.final_logits_bias.to(logits.device)      

    loss = loss_function(logits, trg_real, pad, Loss_Func)
    
    return loss.item()

def create_masks(input_ids, target_ids, pad_token):

    encoder_attention_mask = input_ids.ne(pad_token) 
    decoder_attention_mask = target_ids.ne(pad_token) 
    return encoder_attention_mask.to(device), decoder_attention_mask.to(device)

def train_step(model, src, fading_noise, pad, Loss_Func, channel, opt):

    opt.zero_grad()

    trg_inp = src[:, :-1] 
    trg_real = src[:, 1:]

    src_padding_mask, trg_padding_mask = create_masks(src, trg_inp, pad)
        
    encoder_input = src
      
    Tx = model.encoder(
        input_ids=encoder_input,
        attention_mask=src_padding_mask,
        )   


    channel_enc_inp = model.channel_encoder(Tx)
    fading_signal = channel_Fading(channel_enc_inp, fading_noise, channel) 
    Rx = model.channel_decoder(fading_signal)    
    dec_output = model.decoder(
            input_ids=trg_inp, 
            encoder_hidden_states=Rx, 
            attention_mask=trg_padding_mask, 
            encoder_attention_mask=src_padding_mask,
            )

    logits = model.lm_head(dec_output)
    logits = logits + model.final_logits_bias.to(logits.device) 

    loss = loss_function(logits.contiguous().view(-1, logits.size(-1)), 
                    trg_real.contiguous().view(-1), 
                    pad, 
                    Loss_Func) 

    
    loss.backward()
    opt.step()

    return loss.item()



def valid_loop(args, config, model, test_data, Loss_Func):

    model.eval()
    test_iterator = DataLoader(test_data, batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True, collate_fn=collate_data) 

    Total_loss = [0 for _ in range(len(config.snr_noise_list))]

    with torch.no_grad():       
        for sents in test_iterator:
            sents = sents.to(device)
            for idx, snr in enumerate(config.snr_noise_list):       
                loss = valid_step(model, sents, SNR_to_noise(snr), config.pad_token_id, Loss_Func, args.channel)          
                Total_loss[idx] += loss/len(test_iterator)

    return Total_loss 

def train_loop(args, config, model, train_data, Loss_Func, opt):    

    model.train()
    train_iterator = DataLoader(train_data, batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True, collate_fn=collate_data, shuffle=True)    

    Total_loss = 0

    for sents in train_iterator:
        sents = sents.to(device)
        noise_std = uniform(0.0, 1.0)
        loss = train_step(model, sents, noise_std, config.pad_token_id, Loss_Func, args.channel, opt) 
 
        Total_loss += loss / len(train_iterator)
    
    return Total_loss


class Channels():
    def Pass(self, Tx_sig):
        return Tx_sig

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):

        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)  
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig
        
    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    
    return x

def channel_Fading(Tx_sig, n_var, channel='Rayleigh'):

    channels = Channels()
    
    Tx_sig = PowerNormalize(Tx_sig)

    if channel == 'Pass':
        Rx_sig = channels.Pass(Tx_sig)
    elif channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
    
    return Rx_sig



class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()
        
        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)        
        self.layernorm = nn.LayerNorm(size1, eps=1e-6)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        output = self.layernorm(x1 + x5)

        return output

def SNR_to_noise(snr):
    snr_v = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr_v)
    return noise_std


class GenSC(nn.Module):
    def __init__(self, config):
        super(GenSC, self).__init__()           

        self.shared = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.encoder = Encoder(config, self.shared )
        self.channel_encoder = nn.Sequential(nn.Linear(config.d_model, 256), nn.ReLU(inplace=True), nn.Linear(256, 16))
        self.channel_decoder = ChannelDecoder(16, config.d_model, 512) 
        self.decoder = Decoder(config, self.shared) 
        self.lm_head = nn.Linear(config.d_model, self.shared .num_embeddings, bias=False)  
        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))
         


def greedy_decode(model, src, fading_noise, pad, start_symbol, channel):

    src_padding_mask = src.ne(pad) 

    Tx = model.encoder(
        input_ids=src,
        attention_mask=src_padding_mask,
    )
    
    channel_enc_inp = model.channel_encoder(Tx)
    fading_signal = channel_Fading(channel_enc_inp, fading_noise, channel) 
    channel_dec_output = model.channel_decoder(fading_signal)
        
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
    _, max_len = src.size()
    for i in range(max_len-1):
        trg_padding_mask = outputs.ne(pad)
        dec_output = model.decoder(
            input_ids=outputs, 
            encoder_hidden_states=channel_dec_output, 
            attention_mask=trg_padding_mask, 
            encoder_attention_mask=src_padding_mask,)
        logits = model.lm_head(dec_output)
        logits = logits + model.final_logits_bias.to(logits.device)
        _, next_word = torch.max(logits[: ,-1:, :], dim=-1)
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs[:,1:]


def performance(args, SNR, model, test_iter, chl):

    cos_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
    bleu_score_1gram = BleuScore(1,0,0,0)
    testdata = DataLoader(test_iter, batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True, collate_fn=collate_data)
    StoT = SeqtoText(vocab, Config.eos_token_id)
    bleu, sim = [], []    

    model.eval()
    with torch.no_grad():
        for epoch in range(len(SNR)):
            Tx_word, Rx_word = [], []            
            save_infor = ""
            for snr in tqdm(SNR, desc=f'SNR {chl} - {epoch+1}/{len(SNR)}', total=len(SNR)):
                input_word, word, target_word = [], [], []   
                for iters, sents in enumerate(testdata):
                    sents = sents.to(device)
                    target = sents[:,1:]
                    encoder_input = sents
                    out = greedy_decode(model, encoder_input, SNR_to_noise(snr), Config.pad_token_id, Config.bos_token_id, chl)

                    sentences = out.cpu().numpy().tolist()
                    prediction_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + prediction_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                    if snr in [0, 9, 18] and iters in [0, len(testdata)//2, len(testdata)-1]:                        
                        collect_items = [0, len(prediction_string)//2 , len(prediction_string)-1] 
                        save_infor += f'{snr} - {iters}\n'                       
                        for col_item in collect_items:                       
                            save_infor += 'Predic : ' + prediction_string[col_item] + '\n'
                            save_infor += 'Target : ' + result_string[col_item] + '\n'

                Tx_word.append(word)
                Rx_word.append(target_word)

            with open(args.output + f'Text{epoch}.txt', 'w') as info:
                info.write(save_infor)

            bleu_scores, simi_scores = [], []            

            for sent1, sent2 in zip(Tx_word, Rx_word):
                
                bleu_scores.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) 
                cs_sc = []
                for s1, s2 in zip(sent1, sent2):
                  sw1 = remove_tags(s1)
                  sw2 = remove_tags(s2)
                  embedding_1= cos_model.encode(sw1, convert_to_tensor=True)
                  embedding_2 = cos_model.encode(sw2, convert_to_tensor=True)
                  cs_sc.extend(util.pytorch_cos_sim(embedding_1, embedding_2).cpu().numpy().tolist()[0])
                simi_scores.append(cs_sc)

            bleu_score = np.mean(np.array(bleu_scores), axis=1)
            simi_score = np.mean(np.array(simi_scores), axis=1)
            print(f"BELU SCORE ({epoch+1}) : ", ' '.join(['{:.4f}'.format(x) for x in bleu_score]))
            print(f"SIMI SCORE ({epoch+1}) : ", ' '.join(['{:.4f}'.format(x) for x in simi_score]))
            bleu.append(bleu_score)
            sim.append(simi_score)

    return np.mean(np.array(bleu), axis=0), np.mean(np.array(sim), axis=0)

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def adjust_weights(self):
        weights = (self.w1, self.w2, self.w3, self.w4)
        return tuple(w for w in weights if w != 0.0)

    def compute_blue_score(self, real, predicted):
        adjust_weights = self.adjust_weights()
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, weights=adjust_weights))

        return score

class SeqtoText:
    def __init__(self, vocb_dictionary, eos_token):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.eos_token = eos_token
        
    def sequence_to_text(self, list_of_indices):
        words = []
        for idx in list_of_indices:
            if idx == self.eos_token:
                break
            else:
                word = self.reverse_word_map.get(idx)
                words.append(word)
    
        return ' '.join(words)


def collate_data(batch):

    batch_size = len(batch)
    max_len = max(map(lambda x: len(x), batch))
    sents = np.ones((batch_size, max_len), dtype=np.int64) * Config.pad_token_id
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

    for i, sent in enumerate(sort_by_len):
        length = len(sent)
        sents[i, :length] = sent 

    return  torch.from_numpy(sents)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main(Config):
    def get_args():
        parser = argparse.ArgumentParser(description="Model Training Arguments")
        parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
        parser.add_argument('--channel', type=str, default='Rayleigh', help="Channel type")
        parser.add_argument('--early_stop', type=int, default=15, help="Early stopping criteria")
        parser.add_argument('--checkpoint_path', type=str, default=os.path.join(os.getcwd(), 'pretrained_model/'), help="Path to save checkpoint")
        parser.add_argument('--checkpoint_name', type=str, default='checkpoint.pth', help="Checkpoint file name")
        parser.add_argument('--data_path', type=str, default=os.path.join(os.getcwd(), 'dataset/'), help="Dataset path")
        parser.add_argument('--output', type=str, default=os.path.join(os.getcwd(), 'output/'), help="Output path")
    
        return parser.parse_args()
    
    args = get_args()
    
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    train_data = LoadDataset(args, 'train')
    test_data = LoadDataset(args, 'test')
    
    vocab = json.load(open(args.data_path + 'vocab.json', 'r'))
    Config.vocab_size = len(vocab)
    Config.bos_token_id = vocab["<start>"]
    Config.pad_token_id = vocab['<pad>'] 
    Config.eos_token_id = vocab['<end>']
    Config.msk_token_id = vocab['<mask>']
    
    print("trian max lenght :", max(map(lambda x: len(x), train_data.data)),"test max lenght :", max(map(lambda x: len(x), test_data.data)))
    print("Vab sizes : ", Config.vocab_size, '-', True if len(vocab) == Config.vocab_size else False)
    print("BOS token : ", Config.bos_token_id, '-', True if vocab['<start>'] == Config.bos_token_id else False)
    print("PAD token : ", Config.pad_token_id, '-', True if vocab['<pad>'] == Config.pad_token_id else False)
    print("EOS token : ", Config.eos_token_id, '-', True if vocab['<end>'] == Config.eos_token_id else False)
    print("MSK token : ", Config.msk_token_id, '-', True if vocab['<mask>'] == Config.msk_token_id else False)
    print("Fading Channel : ", args.channel)
    
    model = GenSC(Config).to(device) 
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    
    try :
        pretrain_model_path = os.path.join(args.checkpoint_path, args.checkpoint_name)
        model.load_state_dict(torch.load(pretrain_model_path, weights_only=True))
        info = f'Use pre-train model'
    
    except:
        initNetParams(model)  
        info = f'Model init Net Params'
    
    print(f'{info} with {args.channel} fading channel with parameters {pytorch_total_params}')
    
    Loss_Func = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.AdamW(model.parameters(),1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.75, patience=5)
    
    temp, save_target = 0, [float('inf') for _ in range(len(Config.snr_noise_list))]
    train_list, test_list = [], []
    epoch = 0
    start_time = time.time()
    
    while temp < args.early_stop:
        epoch += 1
        save_model = False
        train_loss = train_loop(args, Config, model, train_data, Loss_Func, optimizer)
        valid_loss = valid_loop(args, Config, model, test_data, Loss_Func)
        train_list.append(train_loss)        
        test_list.append(valid_loss)            
            
        if sum(valid_loss) < sum(save_target) and valid_loss[-1] < save_target[-1]:
            
            temp = 0  
            save_model = True      
    
            for idx, value in enumerate(save_target):
                save_target[idx] = valid_loss[idx]
    
            with open(args.checkpoint_path + args.checkpoint_name , 'wb') as f:
                torch.save(model.state_dict(), f)
                    
        else:
            temp += 1    
            scheduler.step(valid_loss[-1])              
    
        print('Epoch {} - {} times'.format(epoch + 1, temp) + (" ( Model saving ... ) " if save_model else ""))
        print('Valida loss :',' '.join(['{:.4f}'.format(x) for x in valid_loss]))
        print('Saving loss :',' '.join(['{:.4f}'.format(x) for x in save_target]))
            
    end_time = time.time()     
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)   
    
    plt.plot(range(1,len(train_list)+1), train_list)
    plt.plot(range(1,len(test_list)+1), test_list, linestyle="--")
    plt.legend(['Train Loss',"Valid Loss"])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss')
    plt.savefig(args.output+f'Lossimg')
    plt.clf() 
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"           
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.checkpoint_name), weights_only=True)) 
    bleu_score ,similar_score = performance(args, Config.snr_noise_list, model, test_data, args.channel)
    print(f'bleu_score',' '.join(['{:.4f}'.format(x) for x in bleu_score]))
    print(f'simi_score',' '.join(['{:.4f}'.format(x) for x in similar_score]))   
    
    plt.plot(list(range(len(bleu_score.tolist()))), bleu_score.tolist(), marker='o') 
    plt.xlabel('SNR')
    plt.ylabel('BleuScore')
    plt.xticks(list(range(len(bleu_score.tolist()))), Config.snr_noise_list)
    plt.yticks(np.linspace(0,1,11))
    plt.title(f'BleuScore')
    plt.savefig(args.output+f'BleuScore_img')
    plt.clf()
    
    plt.plot(list(range(len(similar_score.tolist()))), similar_score.tolist(), marker='o') 
    plt.xlabel('SNR')
    plt.ylabel('SimilarScore')
    plt.xticks(list(range(len(similar_score.tolist()))), Config.snr_noise_list) 
    plt.yticks(np.linspace(0,1,11))
    plt.title(f'SimilarScore')
    plt.savefig(args.output+f'SimilarScore_img')
    plt.clf()
    
    bleu_score_dict = {}
    bleu_score_dict['Channel'] = args.channel
    bleu_score_dict['BleuScore'] = bleu_score.tolist()
    bleu_score_dict['SimilarScore'] = similar_score.tolist()
    bleu_score_dict['Total_params'] = pytorch_total_params  
    bleu_score_dict['TrainingTime'] = f"{hours} hrs, {minutes} mins, {seconds} secs"
    
    df = pd.DataFrame.from_dict(bleu_score_dict, orient='index')
    df.to_csv(args.output+f'summary.csv')

if __name__ =="__main__":
    main(Config)

