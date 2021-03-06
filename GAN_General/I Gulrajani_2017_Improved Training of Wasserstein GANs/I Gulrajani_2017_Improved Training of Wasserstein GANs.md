# Improved Training of Wasserstein GANs (WGAN-GP)
**Ishaan Gulrajani , Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville**

[ [Paper](https://arxiv.org/abs/1704.00028) ]
[ [Code](https://github.com/igul222/improved_wgan_training.) ]

## 摘要

這篇針對 WGAN 的訓練提出改進，利用 gradient penalty 來限制 disriminator。
作者利用這樣的方式可以成功訓練多種架構的網路，並保有原先 WGAN 的收斂性質。
儘管這篇有比較多的數學推導，但也因此可以合理的解釋設計模型的原因，值得花時間閱讀。

## 將 GAN 改造成 WGAN
1. (WGAN) 訓練多次 discriminator，再訓練 generator
2. (WGAN) 損失函數計算 Wasserstein Loss 而非原先的 JS 散度(discriminator 太容易區分)
3. (WGAN-GP) discriminator 不要使用 batch normalization，可以使用 layer normalization。這是因為 batch normalization 將每個樣本含其他的樣本連結，但在 gradient penalty 的計算上每個樣本都是獨立計算的
4. (WGAN-GP) 移除 discriminator 輸出的 nonlinear activation

<p align="center"><img src="./wgan_gp_algorithm.png" width="600"></p>


## Weight Clipping 的問題
作者利用文中的推論一說明， weight clipping 會導致模型的 weight 傾向極端的分布，也就是接近被 clipping 的值，最終傾向產生簡單的模型，類似正規化 (regularization) 的效果。

<p align="center"><img src="./wgan_vs_wgan_gp.png" width="600"></p>


## Gradient Penalty 的設計
根據 WGAN 的推導，必須要求 discriminator 要是 1-lipschitz function，也就是必須滿足每一點的 gradient norm 都要比 1 小，因此懲罰 gradient 大於 1 的情形(實作上是懲罰不為 1 的情形，作者比較過兩者後認為後者較佳)。另外，計算上只需要考慮到 data distribution 和 generator distribution 中間的點就好，除此之外的點不會在梯度下降的過程中走到。因此可以得到以下在 discriminator 的損失函數

<p align="center"><img src="./wgan_gp_loss.png" width="700"></p>

## WGAN-GP 在不同模型架構上的訓練
作者比較 WGAN, WGAN-GP, DCGAN 在各種不同架構上的訓練，並認為 gradient penalty 的方式可以不容易造成 model collapse，因此容易訓練出結果。作者也另外實驗在離散的輸出上訓練(例如 one-hot vector)，使用 WGAN-GP 也可以訓練。

<p align="center"><img src="./wgan_gp_archtecture_experiment.png" width="600"></p>
<p align="center"><img src="./wgan_gp_archtecture_experiment_result.png" width="600"></p>


## Code
```python
import torch

# 1. Do not use batch normalization in discriminator
# 2. Remove the output activation of discriminator
# 3. Update descriminator several times before updating generator
def descriminator_loss(netD, real, fake, lambda_gp):
    real_score = netD(real)
    fake_score = netD(fake)

    # 1. Randomly choose sample between real and fake
    batch_size = real.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1)
    # unnecessary to backpropagate gradient through fake data
    interp = alpha * real.detach() + (1 - alpha) * fake.detach()
    
    # 2. Calculate dD(x)/dx
    interp.requires_grad = True
    interp_score = netD(interp)
    gradient_outputs = torch.ones(interp_score.size())
    # create_graph and retain_graph must be True, 
    # since we calculate the 2nd gradient when backprogation on gradient penalty
    gradient = torch.autograd.grad(outputs=interp_score, inputs=interp, \
                                    grad_outputs=gradient_outputs, only_inputs=True, \
                                    create_graph=True, retain_graph=True)[0]

    # 3. Gradient Penalty = E[(||dD(x)/dx|| - 1)^2]
    gradient = gradient.view(batch_size, -1)
    gradient_penalty = ((torch.norm(gradient, 2, dim=1) - 1) ** 2).mean()

    return fake_score.mean() - real_score.mean() + lambda_gp * gradient_penalty


def generator_loss(netD, fake):
    fake_score = netD(fake)
    return -fake_score.mean()
```
