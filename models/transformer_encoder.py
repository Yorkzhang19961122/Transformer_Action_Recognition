import torch
from torch import nn
import math
from torch import functional as F
import numpy as np
import torch.utils.data as Data


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerOnlyDecoder(nn.Module):
    def __init__(self, sequence_size, state_size, n_heads, hidden_presentation_dim, d_v, output_sequence_size):
        super(TransformerOnlyDecoder, self).__init__()

        self.n_heads = n_heads
        self.dim_v = d_v
        self.hidden_presentation_dim = hidden_presentation_dim

        # self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(state_size)
        # self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)

        self.sequence_size = sequence_size
        self.state_size = state_size

        # Encoder
        # block one
        # Multi-head attention
        # input size = [batch size, len of sequence, dimension of one state in sequence]
        # the linear layer size [batch size, dimension of one state in sequence, dimension of Q * n_heads]
        self.encoder_Q_matrix_block1 = nn.Linear(in_features=state_size, out_features=d_v * n_heads, bias=False)
        self.encoder_K_matrix_block1 = nn.Linear(in_features=state_size, out_features=d_v * n_heads, bias=False)
        self.encoder_V_matrix_block1 = nn.Linear(in_features=state_size, out_features=d_v * n_heads, bias=False)
        self.encoder_transform_matric_for_context_block1 = nn.Linear(in_features=d_v * n_heads,
                                                                     out_features=hidden_presentation_dim, bias=False)
        # Feed forward
        self.encoder_forward_layer1_block1 = nn.Linear(in_features=hidden_presentation_dim, out_features=hidden_presentation_dim)
        self.encoder_forward_layer2_block1 = nn.Linear(in_features=hidden_presentation_dim, out_features=hidden_presentation_dim)

        # block two
        # Multi-head attention
        self.encoder_Q_matrix_block2 = nn.Linear(in_features=state_size, out_features=d_v * n_heads, bias=False)
        self.encoder_K_matrix_block2 = nn.Linear(in_features=state_size, out_features=d_v * n_heads, bias=False)
        self.encoder_V_matrix_block2 = nn.Linear(in_features=state_size, out_features=d_v * n_heads, bias=False)
        self.encoder_transform_matric_for_context_block2 = nn.Linear(in_features=d_v * n_heads,
                                                                     out_features=hidden_presentation_dim, bias=False)
        # Feed forward
        self.encoder_forward_layer1_block2 = nn.Linear(in_features=hidden_presentation_dim, out_features=hidden_presentation_dim)
        self.encoder_forward_layer2_block2 = nn.Linear(in_features=hidden_presentation_dim, out_features=hidden_presentation_dim)

        self.decoder_output = nn.Linear(in_features=sequence_size * hidden_presentation_dim, out_features=output_sequence_size, bias=False)

    def forward(self, encoder_input_ori):


        # print(encoder_input_ori)
        # inputs shape (batch_size, state_number, state_vector)
        # inputs = [[a1 ... ],
        #           [a2 ... ],
        #           [a3.... ],]

        # first add position msg
        # encoder_inputs = self.src_emb(encoder_input_ori)  # [batch_size, src_len, d_model]
        # print(encoder_input_ori.shape)
        encoder_inputs = self.pos_emb(encoder_input_ori.transpose(0, 1)).transpose(0, 1)
        # print(encoder_inputs.shape)
        """ ____Encoder_____"""
        # ___Multi-head self attention___

        # residual connection
        residual_1 = encoder_inputs


        # calculate for Q , K, V
        # Q = [[q1 ... ],
        #      [q2 ... ],
        #      [q3.... ],]
        # reshape Q to [batch size, n-heads, len of sequence, dimension of q]
        # print(encoder_input_ori.shape)
        Q_en_block1 = self.encoder_Q_matrix_block1(encoder_inputs).\
            view((encoder_inputs.shape[0], self.n_heads, -1, self.dim_v))
        K_en_block1 = self.encoder_K_matrix_block1(encoder_inputs).\
            view((encoder_inputs.shape[0], self.n_heads, -1, self.dim_v))
        V_en_block1 = self.encoder_V_matrix_block1(encoder_inputs).\
            view((encoder_inputs.shape[0], self.n_heads, -1, self.dim_v))

        # K_t = [[. , .  , . ],
        #        [K1, K2 , K3],
        #        [. , .  , . ],
        K_t_en_block1 = K_en_block1.transpose(-1, -2)

        # attention   = [[q1*k1, q1*k2, q1*k3],
        #                [q2*k1, q2*k2, q2*k3],
        #                [q3*k1, q3*k2, q3*k3]]
        attention_en_block1 = torch.matmul(Q_en_block1, K_t_en_block1) / np.sqrt(Q_en_block1.shape[-1])

        # enc_self_attn_mask_ = enc_self_attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # add mask here
        # attention_en_block1.masked_fill_(enc_self_attn_mask_, -1e9)

        # print(attention_en_block1)
        soft_att_en_block1 = nn.Softmax(dim=-1)(attention_en_block1)

        # B = [[b1 ... ],
        #      [b2 ... ],
        #      [b3.... ],]
        B = torch.matmul(soft_att_en_block1, V_en_block1)

        # ___ADD & Norm___
        # dim2 size of B is the same size of q1
        # therefore need to transform here
        # context size [batch size, len of sequence, dimension of each hidden represent]
        context = self.encoder_transform_matric_for_context_block1(B.view(encoder_inputs.shape[0], encoder_inputs.shape[1], -1))
        x = residual_1 + context
        layer_norm = nn.LayerNorm(self.state_size, elementwise_affine=False)(x)

        residual_2 = layer_norm
        x = self.encoder_forward_layer1_block1(layer_norm)
        x = nn.ReLU()(x)
        x = self.encoder_forward_layer2_block1(x)
        output_en_block1 = nn.LayerNorm(self.state_size)(x + residual_2)
        # output of first block of encoder

        # repeat this N times
        # seconde block
        residual_1 = output_en_block1
        Q_en_block2 = self.encoder_Q_matrix_block2(output_en_block1).\
            view((encoder_inputs.shape[0],  self.n_heads, -1, self.dim_v))
        K_en_block2 = self.encoder_K_matrix_block2(output_en_block1).\
            view((encoder_inputs.shape[0],  self.n_heads, -1, self.dim_v))
        V_en_block2 = self.encoder_V_matrix_block2(output_en_block1).\
            view((encoder_inputs.shape[0],  self.n_heads, -1, self.dim_v))
        K_t_en_block2 = K_en_block2.transpose(-1, -2)
        attention_en_block2 = torch.matmul(Q_en_block2, K_t_en_block2) / np.sqrt(Q_en_block2.shape[-1])
        # add mask
        # enc_self_attn_mask_ = enc_self_attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # attention_en_block2.masked_fill_(enc_self_attn_mask_, -1e9)

        soft_att_en_block2 = nn.Softmax(dim=-1)(attention_en_block2)
        B = torch.matmul(soft_att_en_block2, V_en_block2)
        context = self.encoder_transform_matric_for_context_block2(B.view(encoder_inputs.shape[0], encoder_inputs.shape[1], -1))
        x = residual_1 + context
        layer_norm = nn.LayerNorm(self.state_size)(x)
        residual_2 = layer_norm
        x = self.encoder_forward_layer1_block2(layer_norm)
        x = nn.ReLU()(x)
        x = self.encoder_forward_layer2_block2(x)
        output_en_block2 = nn.LayerNorm(self.state_size)(x + residual_2)
        outputs = self.decoder_output(output_en_block2.view(encoder_input_ori.shape[0], -1))

        return outputs