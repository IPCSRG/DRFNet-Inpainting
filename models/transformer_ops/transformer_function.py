"""
2D Vision Transformer class with convolution layer.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in DETR
    * decoder returns a stack of activations from all encoding layers
"""
import copy
import torch
from torch import nn
from einops import rearrange
from models import base_function
from .position_embedding import build_position_embed
import functools



class TransConv(nn.Module):
    def __init__(self, embed_dim,input_nc, inner_nc, num_heads=8, num_layers=2, dim_conv=2048, kernel=3, dropout=0.,
                 activation='gelu', norm='pixel',input_c = 512,patch_size = 16,nor = True, bias=False, norm_layer=nn.BatchNorm2d):
        super(TransConv, self).__init__()
        self.nor = nor
        self.conv_t = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=bias)
        self.norm = norm_layer(inner_nc)
        self.pre_conv = nn.Conv2d(input_c, embed_dim, kernel_size=1, bias=nn.BatchNorm2d)

        self.patch_size = patch_size
        layer = TransformerEncoderLayer(embed_dim, num_heads, dim_conv, kernel, dropout, activation, norm)
        self.layers = _get_clones(layer, num_layers)

        self.out_conv = nn.Conv2d(embed_dim, input_c, kernel_size=1, padding=0, bias=nn.BatchNorm2d)

    def forward(self, src, src_key_padding_mask=None, src_mask=None, pos=None, bool_mask=True):

################conv_in
        src = self.conv_t(src)
        if self.nor:
            src = self.norm(src)
        b, c_o, h_0, w_0 = src.size()
        h_0, w_0 = int(h_0 / self.patch_size), int(w_0 / self.patch_size)
        src = rearrange(src, 'b c (h p1) (w p2) ->b (c h w) p1 p2', p1=self.patch_size, p2=self.patch_size)
        src = self.pre_conv(src)


        out = src
        # outs = []
        src_key_padding_mask_bool = src_key_padding_mask
        for i, layer in enumerate(self.layers):
            if src_key_padding_mask is not None:
                src_key_padding_mask_bool = src_key_padding_mask < 0.5 if bool_mask else src_key_padding_mask
                src_key_padding_mask = src_key_padding_mask ** 0.5
            out = layer(out, src_key_padding_mask_bool, src_mask, pos)
            # outs.append(out)
        out = self.out_conv(out)
        out = rearrange(out, 'b (c h w) p1 p2 ->b c (h p1) (w p2)', c=c_o, h=h_0, w=w_0, p1=self.patch_size, p2=self.patch_size)
        return out




######################################################################################
# base transformer operation
######################################################################################
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dim_conv=2048, kernel=3, dropout=0., activation='gelu', norm='pixel' ,input_c = 512):
        """
        Encoder transformer block
        :param embed_dim: total dimension of the model
        :param num_heads: parallel attention heads
        :param dim_conv: feature in feedforward layer
        :param kernel: kernel size for feedforward operation, kernel=1 is similar to MLP layer
        :param dropout: a dropout layer on attention weight
        :param activation: activation function
        :param norm: normalization layer
        """
        super(TransformerEncoderLayer, self).__init__()


        self.attn = MultiheadAttention(embed_dim, num_heads, dropout)
################conv_mlp
        self.conv1 = nn.Conv2d(embed_dim, dim_conv, kernel_size=kernel, padding=int((kernel - 1) / 2))
        self.conv2 = nn.Conv2d(dim_conv, embed_dim, kernel_size=1, padding=0, bias=nn.BatchNorm2d)
        # self.relu2 = nn.ReLU(True)
        self.norm1 = base_function.PixelwiseNorm(embed_dim)
        self.norm2 = base_function.PixelwiseNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)


    def _with_pos_embed(self, x, pos=None):
        return x if pos is None else x + pos

    def forward(self, src, src_key_padding_mask=None, src_mask=None, pos=None):

        b, c, h, w = src.size()
        src2 = self.norm1(src)
        src2 = rearrange(src2, 'b c h w->b (h w) c')
        q = k = self._with_pos_embed(src2, pos)
        src2 = self.attn(q, k, src2, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        src2 = rearrange(src2, 'b (h w) c->b c h w', h=h, w=w)
        src = src + self.dropout(src2)
        src2 = self.norm2(src)
        src2 = self.conv2(self.dropout(self.activation(self.conv1(src2))))
        src = src + self.dropout(src2)

        return src





class MultiheadAttention(nn.Module):
    """Allows the model to jointly attend to information from different position"""
    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.bias = bias
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_out = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        if self.bias:
            nn.init.constant_(self.to_q.bias, 0.)
            nn.init.constant_(self.to_k.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        b, n, c, h = *q.shape, self.num_heads
        # calculate similarity map
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q = rearrange(q, 'b n (h d)->b h n d', h=h)
        k = rearrange(k, 'b n (h d)->b h n d', h=h)
        v = rearrange(v, 'b n (h d)->b h n d', h=h)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # assign the attention weight based on the mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            if key_padding_mask.dtype == torch.bool:
                dots = dots.masked_fill(key_padding_mask, float('-inf'))
            else:
                dots = torch.where(dots > 0, key_padding_mask * dots, dots/(key_padding_mask+1e-5))
        # calculate the attention value
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        # projection
        out = torch.einsum('bhij, bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])