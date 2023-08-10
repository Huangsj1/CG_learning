# GAN 背景

**GAN生成对抗网络**：包括生成器generator和判别器discriminator，通过输入随机生成的向量（最好是高斯分布）输入到生成器中，其目的是生成接近于真实（接近于train训练集）的图像，而判别器目的是辨别出图像是否真实（是否符合训练集）。可以理解为生成器是将符合某种分布的随机输入**映射**到符合真实图像的分布（将train中的图像理解为高维的分布），而判别器是判断随机数据映射后的分布与真实分布的**拟合程度**

# Generative Latent Optimization（GLO）

**Generative Latent Optimization（GLO）生成式潜在优化**：能够将低维的 latent code 与高维的结果（例如图像）建立联系（例如从latent code中经过decoder还原出图像）

GLO的具体步骤：

1. 对于N张数据集里面的图像x，用正态分布随机初始化N个d维的latent code记为z，将每个 z 与 x 随机配对 $\{\{z_1,x_1\}...\{z_n,x_n\}\}$
2. [论文中](https://arxiv.org/abs/1707.05776)选择了GAN的生成器作为本方法的decoder（这是一个神经网络）
3. 将 z 输入到decoder中，得到一张图像，将这张图像与该 z 对应的真实图像作loss，反向传播，**同时优化 z 和 decoder**（不同于传统的神经网络单纯优化decoder，也不同于可微渲染中单纯优化latent code，这里把两个都放到opeimizer中同时优化（只需要为输入 z 和神经网络decoder的参数都设置为需要进行梯度下降，并放到optimizer中更新即可））
4. 更新完 z 后，将其投影回单位球（通过除以 $max(||z||_2,1)$）来实现，这样原本在单位球内符合正态分布的点，经过训练后在单位球内满足了某种分布（可以经过decoder映射到真实图像的分布）

下面是decoder中用到的DCGAN的生成器

![[Pasted image 20230808163415.png]]
