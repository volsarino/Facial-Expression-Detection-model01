import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self,in_channels=3,patch_size=16,emb_size=768,img_size=224):
        super().__init__()
        self.patch_size=patch_size
        self.n_patches=(img_size//patch_size)**2
        self.projection=nn.Sequential(nn.Conv2d(in_channels,emb_size,kernel_size=patch_size,stride=patch_size),
                                      Rearrange('b e h w ->b( h w ) e '))
        
    def forward(self,x):
        x=self.projection(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self,embed_dim=192,num_heads=3,mlp_ratio=4.0):
        super().__init__()
        self.norm1=nn.LayerNorm(embed_dim)
        self.attn=nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.norm2=nn.LayerNorm(embed_dim)
        hidden_dim=int(embed_dim*mlp_ratio)
        self.mlp=nn.Sequential(nn.Linear(embed_dim,hidden_dim),
                               nn.GELU(),
                               nn.Linear(hidden_dim,embed_dim))

    def forward(self,x):
        norm_x=self.norm1(x)
        attn_out,_=self.attn(norm_x,norm_x,norm_x)
        x=x+attn_out
        x=x+self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self,img_size=224,patch_size=16,in_channels=3,num_classes=7,embed_dim=192,depth=4,num_heads=3):
        super().__init__()
        self.patch_embed=PatchEmbedding(in_channels=in_channels,
                                        patch_size=patch_size,
                                        emb_size=embed_dim,
                                        img_size=img_size)
        self.num_patches=self.patch_embed.n_patches

        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed=nn.Parameter(torch.zeros(1,self.num_patches+1,embed_dim))
        self.blocks=nn.ModuleList([TransformerEncoderBlock(embed_dim,num_heads)for _ in range(depth)])
        self.norm=nn.LayerNorm(embed_dim)
        self.head=nn.Linear(embed_dim,num_classes)

    def forward(self,x):
        B=x.shape[0]
        x=self.patch_embed(x)
        cls_tokens=self.cls_token.expand(B,-1,-1)
        x=torch.cat((cls_tokens,x),dim=1)
        x=x+self.pos_embed
        for block in self.blocks:
            x=block(x)

        x=self.norm(x)
        cls_out=x[:,0]
        logits=self.head(cls_out)
        return logits

if __name__ == "__main__":
    model = ViT(num_classes=7)
    dummy_input = torch.randn(1, 3, 224, 224)  # バッチサイズ1, 224x224 RGB画像
    output = model(dummy_input)
    print("モデル出力形状 (バッチサイズ, 感情クラス数):", output.shape)
