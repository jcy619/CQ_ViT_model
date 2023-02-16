"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from matcher import build_matcher,HungarianMatcher

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth随机深度) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout（另一种形式的dropout） in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth随机深度) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW] [8,768,14,14]->[8,768,196]
        # transpose: [B, C, HW] -> [B, HW, C]   [8,196,768]

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# class QKV_Attention(nn.Module):
#     def __init__(self,img_size=32,dim=128,batch_size=8,patch_size=4,num_heads=8,qkv_bias=False,qk_scale=None,dropout=0.1):
#         super().__init__()
#         self.batch_size=batch_size
#         self.patch_size=patch_size
#         self.dim = dim
#
#         self.num_patch = int(img_size/patch_size)**2
#         self.num_heads = num_heads
#         self.head_dim = dim//num_heads
#         self.scale = qk_scale or self.head_dim**-0.5
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim,dim)
#         self.proj_drop = nn.Dropout(dropout)
#
#     def forward(self,x):
#
#         B,N,C = x["query"].shape
#         B1,N1,C1 = x["key"].shape
#
#
#         q = x["query"].reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3) #(8,12,1,64)
#         k = x["key"].reshape(B1,N1,self.num_heads,self.head_dim).permute(0,2,1,3) #(8,12,196,64)
#         v = x["value"].reshape(B1,N1,self.num_heads,self.head_dim).permute(0,2,1,3)
#
#         attn = (q @ k.transpose(-2,-1)) *self.scale
#         attn = attn.softmax(dim=-1)
#         x = (attn@v).transpose(1,2).reshape(B,N,self.dim)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self,img_size=32,patch_size=4,batch_size=8,dropout=0.,decoder_embed_dim=128,
                 num_heads=8,mlp_ratio = 4.0,depth_i = 0):
        super().__init__()
        #两个多头注意力
        self.batch_size=batch_size
        self.i = depth_i
        self.patch_num=int(img_size/patch_size)**2
        self.decoder_embed_dim = decoder_embed_dim


        self.self_attn = nn.MultiheadAttention(decoder_embed_dim,num_heads,dropout)

        self.multihead_attn = nn.MultiheadAttention(decoder_embed_dim,num_heads,dropout)
        #feed forward部分
        dim_feedforward = int(mlp_ratio*decoder_embed_dim)
        self.linear1 = nn.Linear(decoder_embed_dim,dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward,decoder_embed_dim)

        self.norm1 = nn.LayerNorm(decoder_embed_dim)
        self.norm2 = nn.LayerNorm(decoder_embed_dim)
        self.norm3 = nn.LayerNorm(decoder_embed_dim)


        self.dropout1 =nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()


    def forward(self,x):

        tgt = x["tgt"]
        memory = x["x"]
        pos = x["pos"]
        query_pos = x["query_pos"]

        q = k = tgt+query_pos
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=None, key_padding_mask=None)[0]  # (10,bs,dim)

        tgt = tgt + self.dropout1(tgt2)  # (10,bs,dim)

        tgt = self.norm1(tgt)
        # query = tgt+query_pos #(10,bs,dim)
        query = tgt+query_pos
        key = memory+pos  # (65,bs,dim)
        tgt2 = self.multihead_attn(query=query,
                                   key=key,
                                   value=memory, attn_mask=None,
                                   key_padding_mask=None)[0]  # (10,bs,dim)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        output = {}
        output["tgt"] = tgt
        output["x"] = memory
        output["pos"] = pos
        output["query_pos"] = query_pos

        return output  # (10,bs,256)

        # if self.i == 0:
        #     tgt = x["tgt"]
        #     memory = x["x"]
        #     pos = x["pos"]
        #     query_pos = x["query_pos"]
        #
        #     q = k = tgt + query_pos
        #     tgt2 = self.self_attn(q, k, value=tgt, attn_mask=None, key_padding_mask=None)[0]  # (10,bs,dim)
        #
        #     tgt = tgt + self.dropout1(tgt2)  # (10,bs,dim)
        #
        #     tgt = self.norm1(tgt)
        #     # query = tgt+query_pos #(10,bs,dim)
        #     query = tgt
        #     key = memory + pos  # (65,bs,dim)
        #     tgt2 = self.multihead_attn(query=query,
        #                                key=key,
        #                                value=memory, attn_mask=None,
        #                                key_padding_mask=None)[0]  # (10,bs,dim)
        #     tgt = tgt + self.dropout2(tgt2)
        #     tgt = self.norm2(tgt)
        #     tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        #     tgt = tgt + self.dropout3(tgt2)
        #     tgt = self.norm3(tgt)
        #     output = {}
        #     output["tgt"] = tgt
        #     output["x"] = memory
        #     output["pos"] = pos
        #     output["query_pos"] = query_pos
        #
        #     return output  # (10,bs,256)
        # else:
        #
        #     tgt = x["tgt"]
        #     memory = x["x"]
        #     pos = x["pos"]
        #     query_pos = x["query_pos"]
        #
        #     q = k = tgt
        #     tgt2 = self.self_attn(q, k, value=tgt, attn_mask=None, key_padding_mask=None)[0]  # (10,bs,dim)
        #
        #     tgt = tgt + self.dropout1(tgt2)  # (10,bs,dim)
        #
        #     tgt = self.norm1(tgt)
        #     # query = tgt+query_pos #(10,bs,dim)
        #     query = tgt
        #     key = memory  # (65,bs,dim)
        #     tgt2 = self.multihead_attn(query=query,
        #                                key=key,
        #                                value=memory, attn_mask=None,
        #                                key_padding_mask=None)[0]  # (10,bs,dim)
        #     tgt = tgt + self.dropout2(tgt2)
        #     tgt = self.norm2(tgt)
        #     tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        #     tgt = tgt + self.dropout3(tgt2)
        #     tgt = self.norm3(tgt)
        #     output = {}
        #     output["tgt"] = tgt
        #     output["x"] = memory
        #     output["pos"] = pos
        #     output["query_pos"] = query_pos
        #
        #     return output  # (10,bs,256)


class ViTDecoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_c=3, num_classes=10,
                 embed_dim=512, depth=4,depth_decoder=4,decoder_embed_dim=512, encoder_num_heads=8,decoder_num_heads=8, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None,drop_ratio=0.,attn_drop_ratio=0.,num_queries=10, embed_layer=PatchEmbed, norm_layer=None,act_layer = None,
                 ):

        super().__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim


        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) #[1,197,768]

        self.pos_drop = nn.Dropout(p=drop_ratio)



        #encoder部分
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=0,#drop_path_ratio=dpr[i]
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
#-----------decoder部分----------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.num_queries = num_queries
        self.query_embed=nn.Embedding(num_queries,decoder_embed_dim)


        self.decoder_blocks = nn.Sequential(*[
            TransformerDecoderLayer(decoder_embed_dim=decoder_embed_dim,num_heads=decoder_num_heads,
                                    mlp_ratio=mlp_ratio,dropout=0.,depth_i = i)
            for i in range(depth_decoder)
        ])


        self.norm = norm_layer(embed_dim)

        # Representation layer
        self.has_logits = False
        self.pre_logits = nn.Identity()#没有实质性的操作，不改变输入，直接输出input，占位用的

        # Classifier head(s) num_features是class的embed维度
        self.class_embed = nn.Linear(decoder_embed_dim,num_classes)
        # Weight init
        #正太分布初始化，mean = 0，std = 0.02
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)


    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 64, 512]
        # [1, 1, 512] -> [B, 1, 512]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed) #dropout
        x = self.blocks(x)
        x = self.norm(x) #（8，65，512）
        return x





    def forward(self, x):
        bs = x.shape[0]
        x = self.forward_features(x) #(bs,65,512)
        x = self.decoder_embed(x).permute(1,0,2) #(bs,65,decoder_dim)--(65,bs,128)

        #x = x+self.decoder_pos_embed
        decoder_pos_embed = self.decoder_pos_embed.permute(1,0,2).repeat(1,bs,1)#(65,bs,512)
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1,bs,1)#(10,bs,128)
        tgt = torch.zeros_like(query_embed)
        input = {}
        input["tgt"] = tgt
        input["x"] = x
        input["pos"] = decoder_pos_embed
        input["query_pos"] = query_embed
        hs = self.decoder_blocks(input) #(10,bs,dim)
        hs = hs["tgt"].transpose(0,1) #(bs,10,dim)
        outputs_class = self.class_embed(hs)#(bs,query_num,num_class)

        return outputs_class


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    # 返回input中指定维度中k个最大元素。dim默认选择最后一维。 largest=False返回k个最小元素.soreted是否排序；返回（值，索引）
    _, pred = output.topk(maxk, 1, True, True)#（6，1）
    pred = pred.t()#（1，6）
    correct = pred.eq(target.view(1, -1).expand_as(pred)) #(F,F,F,F,F,F)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size)) #*100/batch_size,所有预测正确的平分到各个query中

    return res


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        #计算真实box和模型输出的饥饿分配
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict,eos_coef,losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting（不考虑） the special no-object category，
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category（0.1）
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight',empty_weight)



    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        #assert 'pred_logits' in outputs
        src_logits = outputs #（bs，q，cls）
        #label_id,query_id        idx = self._get_src_permutation_idx(indices)
        idx = self._get_src_permutation_idx(indices)
        #targets=targets
        target_classes_o = targets

        #print("target_classes_o",target_classes_o.device)
        #定义一个大小为（bs，num_queries），元素全为10
        target_classes = torch.full(src_logits.shape[:2], self.num_classes-1,
                                    dtype=torch.int64, device=src_logits.device) #(2,100)
        #print("target_classes", target_classes.device)

        target_classes[idx] = target_classes_o#(bs,num_query)。每个batch中，query学习到的是哪个obj.query14学习到的是obj82

        #这个batch中类损失
        #(2,92,100) (2,100)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes,self.empty_weight)

        losses = {'loss_ce': loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            #(6,92) (6,)
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] #错误评分。100基准。一个没预测对，100

        temp = src_logits[idx]
        pred = torch.max(src_logits[idx],dim=1)[1]  # （6，1）
        losses['predict'] = pred
        losses['logits']=temp #为测试用
        losses['idx'] = idx  #为测试用

        #loss_ce,预测分类的损失函数，class_error预测错误个数评估
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        #(6,)[0,0,1,1,1,1] 0表示第1个batch，1表示第二个batch
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        #(6,)[14,34,18,30,...]学习到obj的query
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = outputs #(bs,num_query,num_class)


        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets) #哪个query学到了label，学到了几个，0个表示1个

        # Compute all the requested losses
        losses = {}

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))
        #loss
        return losses



def dert_model_flower(args):
    num_classes=args.num_classes
    device = torch.device(args.device)
    model = ViTDecoder(img_size = 224,
                       patch_size=16,
                       num_classes=num_classes,
                       embed_dim = 768,
                       depth=6,
                       depth_decoder=6,
                       decoder_embed_dim=768,
                       encoder_num_heads=12,
                       decoder_num_heads=12,
                       num_queries = num_classes-1,#查询有价值的类

                       )

    matcher = build_matcher(args)
    matcher.to(device)
    weight_dict = {"loss_ce":1}
    losses = ['labels']
    eos_coef = 0.1
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,eos_coef=eos_coef,losses=losses)
    #criterion.to(device)
    return model,criterion