## DAMA - Student Collaboration Improves Self-Supervised Learning: Dual-Loss Adaptive Masked Autoencoder for Brain Cell Image Analysis

This is a PyTorch/GPU implementation of the paper [Student Collaboration Improves Self-Supervised Learning: Dual-Loss Adaptive Masked Autoencoder for Brain Cell Image Analysis](Arxiv link goes here):

<p align="center">
  <img src="https://github.com/hula-ai/DAMA/blob/main/imgs/ECCV-pipeline.png" width="720">
</p>

* This repo is based on PyTorch=1.10.1 and timm=0.5.4

Below is the fine-tune result of DAMA compared to other state-of-the-art methods pretrained on brain cells dataset and ImageNet-1k. Please see the paper for detailed results.

### Brain Cell datasets
Manually collected set *Aug-30k* and noisy set *Real-30k*
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th rowspan="2">Methods</th>
<th rowspan="2">Pretrained Epochs</th>
<th colspan="2">Pretrained Sets</th>
<tr>
<td align="center">Aug-30k</td>
<td align="center">Real-30k</td>
</tr>
<!-- TABLE BODY -->
<tr>
<td align="left">Rand. init</td>
<td align="center">300</td>
<td colspan="2">74</td>
</tr>
<tr>
<td align="left">Moco-v3</td>
<td align="center">500</td>
<td align="center">73.75</td>
<td align="center">77.19</td>
</tr>
<tr>
<td align="left">MAE</td>
<td align="center">500</td>
<td align="center">66.69</td>
<td align="center">67.25</td>
</tr>
<tr>
<td align="left">Data2Vec</td>
<td align="center">800</td>
<td align="center">73.12</td>
<td align="center">67.94</td>
</tr>
<tr>
<td align="left">Data2Vec</td>
<td align="center">1600</td>
<td align="center">76.31</td>
<td align="center">73.69</td>
</tr>
<tr>
<td align="left">DAMA<sub>rand</sub></td>
<td align="center">500</td>
<td align="center">76.96</td>
<td align="center">76.38</td>
</tr>
<tr>
<td align="left">DAMA<sub>adap</sub></td>
<td align="center">500</td>
<td align="center">78.19</td>
<td align="center">78.12</td>
</tr>
</tbody></table>

### Pretrained on ImageNet-1k with ViT-Base
Due to computational resource, DAMA is trained **only once** without any ablation experiment for ImageNet and with similar configuration as for trained the brain cell dataset.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<tr>
    <th>Methods</th>
    <th>Pretrained Epochs</th>
    <th>Acc</th>
</tr>
<!-- TABLE BODY -->
<tr>
<td align="left">Moco-v3</td>
<td align="center">600</td>
<td align="center">83.2</td>
</tr>
<tr>
<td align="left">BEiT</td>
<td align="center">800</td>
<td align="center">83.4</td>
</tr>
<tr>
<td align="left">SimMIM</td>
<td align="center">800</td>
<td align="center">83.8</td>
</tr>
<tr>
<td align="left">Data2Vec</td>
<td align="center">800</td>
<td align="center">84.2</td>
</tr>
<tr>
<td align="left">DINO</td>
<td align="center">1600</td>
<td align="center">83.6</td>
</tr>
<tr>
<td align="left">iBOT</td>
<td align="center">1600</td>
<td align="center">84.0</td>
</tr>
<tr>
<td align="left">MAE</td>
<td align="center">1600</td>
<td align="center">83.6</td>
</tr>
<tr>
<td align="left">DAMA</td>
<td align="center">500</td>
<td align="center">83.17</td>
</tr>    
</tbody></table>

### Pre-training DAMA
```
python submitit_pretrain.py --arch main_vit_tiny \
      --batch_size 64 --epochs 500 --warmup_epochs 40 \
      --mask_ratio 0.8 --mask_overlap_ratio 0.5 --last_k_blocks 6 --norm_pix_loss \
      --data_path path_to_dataset_folder \
      --job_dir path_to_output_folder \
      --nodes 1 --ngpus 4
```

### Fine-tuning DAMA
```
python submitit_finetune.py --arch main_vit_tiny \
      --batch_size 128 --epochs 150  \
      --data_path path_to_dataset_folder \
      --finetune path_to_pretrained_file \
      --job_dir path_to_output_finetune_folder \
      --dist_eval --nodes 1 --ngpus 4
```
