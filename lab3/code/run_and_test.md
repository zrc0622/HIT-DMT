# 步骤
1. 下载CIFAR-100数据集
2. 模型finetune
4. 测试成功率：0.8992


# 代码
## test
```
bash run_scripts/zeroshot_eval.sh 0 /home/lsy/Projects/CLIP/Data flowers2 ViT-B-16 RoBERTa-wwm-ext-base-chinese /home/lsy/Projects/CLIP/Data/pretrained_weights/clip_cn_vit-b-16_finetune_cifar-100.pt
```
```
bash run_scripts/zeroshot_eval.sh 0 /home/lsy/Projects/CLIP/Data cifar-100 ViT-B-16 RoBERTa-wwm-ext-base-chinese /home/lsy/Projects/CLIP/Data/pretrained_weights/clip_cn_vit-b-16_finetune_cifar-100.pt
```
## finetune
```
bash run_scripts/flickr30k_finetune_vit-b-16_rbt-base-mine.sh /home/lsy/Projects/CLIP/Data
```

# 结果
## finetune
### cifar100
```
torch.Size([10000, 100])
Result:
zeroshot-top1: 0.8992
Finished.
```
### flower1
```
torch.Size([9, 4])
Result:
zeroshot-top1: 1.0
Finished.
```
### flower2
```
torch.Size([500, 5])
Result:1
zeroshot-top1: 0.93
Finished.
```