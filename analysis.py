import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
#from vit_model import vit_base_patch16_224_in21k as create_model
from dert_model import dert_model_flower as create_model

import math
import numpy as np

from PIL import Image
import requests
import matplotlib.pyplot as plt
from thop import profile
import ipywidgets as widgets
from IPython.display import display, clear_output


from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from torch.nn.functional import dropout,linear,softmax
torch.set_grad_enabled(False)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485,0.456,0.406], [0.299,0.224,0.225])])

    # load image
    img_path = "image/flower.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    im = Image.open(img_path).convert("RGB")
    plt.imshow(im)
    # [N, C, H, W]
    img = data_transform(im)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    labels = torch.tensor([4])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)


    model,criterion = create_model(args)
    # load model weights
    model_weight_path = "./model-199.pth"
    weights_dict = torch.load(model_weight_path,map_location=device)
    model.load_state_dict(weights_dict,strict=False)
    model.eval()
    criterion.eval()
    CLASSES=["daisy","dandelion","roses","sunflowers","tulips","null"]
    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


    for name,parameters in model.named_parameters():
        # 获取训练好的object queries，即pq:[5,768]
        if name == 'query_embed.weight':
            pq = parameters
        # 获取解码器的最后一层的交叉注意力模块中q和k的线性权重和偏置:[2304,768],[2304]
        #if name == 'decoder_blocks.5.multihead_attn.in_proj_weight':
        if name == 'decoder_blocks.5.multihead_attn.out_proj.weight':
            in_proj_weight = parameters
        if name == 'decoder_blocks.5.multihead_attn.in_proj_bias':
            in_proj_bias = parameters



        # predict class
    outputs = model(img.to(device))
    loss_dict = criterion(outputs, labels)
    probs = outputs.softmax(-1)[0,:,:-1] #(num_queries,num_classes(去掉空类))
    query_id = loss_dict['idx'][1]
    predict_cla = (loss_dict['predict']).numpy()
    predict = torch.softmax(loss_dict["logits"], dim=1).squeeze()
    logits = predict[predict_cla[0]].numpy()
    print(logits)
    print("query_id", query_id)
    print("labels", labels)



    conv_features,enc_attn_weights, dec_attn_weights = [], [], []
    cq = []  # 存储detr中的 cq
    pk = []  # 存储detr中的 encoder pos/decoder pos
    memory = []  # 存储encoder的输出特征图memory
    #print(model)
    # 注册hook

    hooks = [
        # # 获取resnet最后一层特征图
        # model.backbone[-2].register_forward_hook(
        #     lambda self, input, output: conv_features.append(output)
        # ),
        # 获取encoder的图像特征图memory
        model.blocks.register_forward_hook(
            lambda self, input, output: memory.append(output)
        ),
        #获取encoder的最后一层layer的self-attn weights
        model.blocks[-1].attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output)
        ),
        # 获取decoder的最后一层layer中交叉注意力的 weights
        model.decoder_blocks[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
        # 获取decoder最后一层self-attn的输出cq
        model.decoder_blocks[-1].norm1.register_forward_hook(
            lambda self, input, output: cq.append(output)
        ),
        # #获取图像特征图的位置编码pk,query的位置代码
        # model.query_embed.register_forward_hook(
        #     lambda self, input, output: pk.append(output)
        # ),
    ]
    outputs = model(img)


    for hook in hooks:
        hook.remove()
    # don't need the list anymore

    enc_attn_weights = enc_attn_weights[0]  # [1,197,768]   : [N,L,S]
    dec_attn_weights = dec_attn_weights[0]  # [1,5,197]   : [N,L,S] --> [batch, tgt_len, src_len]
    memory = memory[0].transpose(0,1)  # [1,197,768]

    cq = cq[0]  # decoder的self_attn:最后一层输出[5,1,768]
    #pk = pk[0]位置编码

    pq = pq.unsqueeze(1).repeat(1,1,1) #(5,768)
    q = pq+cq #cq(5,1,768)
    k = memory #(197,1,768)

    # 将q和k完成线性层的映射，代码参考自nn.MultiHeadAttn()
    _b = in_proj_bias
    _start = 0
    _end = 256
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    q = linear(q, _w, _b)

    _b = in_proj_bias
    _start = 256
    _end = 256 * 2
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    k = linear(k, _w, _b)

    scaling = float(256) ** -0.5
    q = q * scaling
    q = q.contiguous().view(5, 8, 32).transpose(0, 1) #(8,5,32)
    k = k.contiguous().view(-1, 8, 32).transpose(0, 1) #(8,197,32)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2)) #(8,5,197)

    attn_output_weights = attn_output_weights.view(1, 8, 5, 197)
    attn_output_weights = attn_output_weights.view(1 * 8, 5, 197)
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = attn_output_weights.view(1, 8, 5, 197)

    # 后续可视化各个头
    attn_every_heads = attn_output_weights  # [1,8,100,850]
    attn_outputs_weights = attn_output_weights.sum(dim=1) / 8  # [1,100,850]

    # -----------#
    #   可视化
    # -----------#
    # get the feature map shape
    h, w = 16,16

    fig, axs = plt.subplots(ncols=5, nrows=10, figsize=(22, 28))  # [11,2]
    colors = COLORS * 100



    # 可视化

    for idx,ax_i in zip(range(5),axs.T):
        ax = ax_i[0]
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(f"{CLASSES[probs[query_id].argmax()]}, {logits:.3}", fontsize=30)


        # 可视化框和类别
        ax = ax_i[1]
        ax.imshow(dec_attn_weights[0, idx][1:].view(14, 14))
        ax.axis('off')
        ax.set_title(f'query id: {idx}', fontsize=30)

        # 分别可视化8个头部的位置特征图
        for head in range(2, 2 + 8):
            ax = ax_i[head]
            ax.imshow(attn_every_heads[0, head - 2, idx][1:].view(14, 14))
            ax.axis('off')
            ax.set_title(f'head:{head - 2}', fontsize=30)




    fig.tight_layout()  # 自动调整子图来使其填充整个画布
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)  # 第5类是空类
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/home/featurize/data/cifar-10/sample") #******
    parser.add_argument('--data-path', type=str,
                        default="")  # ******
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)


