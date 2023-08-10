# NeRF in the wild

## 1. motivation 动机

NeRF只能用在照片中**光线影响都是constant**的以及**场景的内容是静态的**情况下，即对于variable illumination可变光 和 transient occluders 瞬态障碍处理的不好

## 2. 改进方法

1. model每张图片的辐射变量，如曝光、光照、天气、色调等在一个**基于学习的低维的 latent space**中，这样就可以控制output的appearance
2. model整个场景作为union of shared 和 image-dependent elements，这样就可以**解构 “static” 和 “transient” 两部分**，这样就可以只渲染“static”的部分

![[Pasted image 20230808191900.png|400]]

### 1. Latent Appearance Modeling 潜在外观建模

位置信息position **x** 依然作为同NeRF前面一样输入到MLP中得到static的体密度$\sigma_{static}$ ，但是在网络的后面部分除了原本的view direction **d** 之外，还加入了 $l^{(a)}$ 作为appearance embedding来学习**每一张图片的辐射亮度**（光照、色调等）。

因为不同的图片有着不同的风格，所以上面的 $l^{(a)}$ 对应的是每一张图片的 $l^{(a)}_i$ ，同时每张图片在相同采样点、像素点的color也不同，于是从采样点的 $\textbf{c}$ 变成 $\textbf{c}_i$，像素点的 $\hat{\textbf{C}}(\textbf{r})$ 变成了 $\hat{\textbf{C}}_i(\textbf{r})$

![[Pasted image 20230808193041.png|400]]

因为 $l_i^{(a)}$ 是 image-dependent 的，所以 $l_i^{(a)}$ 能够学到的是不同图片各自的特点（通用的几何被 $\sigma$ 学走）；而且 $l_i^{(a)}$ 是和 **d** 一起在后面输入到网络中的，这里的特点又和 **d** 有关（**d**学的是radiance filed，与光照、颜色有关），所以 **$l_i^{(a)}$ 学到的是每张图片特有的辐射亮度特征**，同时仍然能保证场景信息是静态的且能够共享

#### latent code 的应用

因为latent code $l^{(a)}$ 学的是每张图片的特有的辐射信息，所以可以通过对两张图片的辐射信息进行**插值得到渐变过程**

![[Pasted image 20230809115737.png]]

### 2. Transient Objects 瞬态物体现象

#### 1. transient network 瞬态网络

**原因**：每张图片除了特有的辐射亮度，还都有特定的噪声/瞬态物体

**做法**：将原本的NeRF中的网络结构视作static静态网络（[[#1. Latent Appearance Modeling 潜在外观建模|上面]]讲的），同时多添加了一个transient瞬态网络，其结构与static类似（前面部分相同，为共有的），但是后面部分添加了 $l_i^{(t)}$ 来作为transient embedding来学习**每张图片特有的瞬态物体信息**，包括几何和辐射亮度（因为这个transient网络最终输出的有 $\sigma_i^{(\tau)}(t)$ 和 $c_i^{(\tau)}(t)$，它们结合网络中间 / 共享部分输出的 $\sigma_i(t)$ 和 $c_i(t)$ 来得到每个像素最终的颜色 $\hat C_i(r)$，这个最终颜色渲染出来的是包含static静态物体和transient瞬态物体的场景；而前面网络的输出 $\sigma_i(t)$ 是用来构建static静态物体场景的，这也就表面transient网络最终的输出 $\sigma^{(\tau)}_i(t)$是场景中的瞬态物体信息，而 $l_i^{(t)}$ 学习的也是瞬态物体信息，包括几何和radiance field）

![[Pasted image 20230809092806.png]]

#### 2. 在瞬态网络中引入不确定度 uncertainty

**原因**：因为每一张图片都有噪声（就是transient），这些噪声各不相同，**模型很难够准确地学习到每张图片的所有噪声**，所以这里就引入了不确定度 uncertainty，对于这些噪声部分可以不用预测的非常准（像素值不用太准确），对于**预测的不是很准的地方只要有较高的不确定度就够**了（对于不确定度高的部分，预测值和真实值相差大一点损失也不会太大）

**做法**：于是transient网络中最后的输出除了 噪声物体体密度 $\sigma_i^{(\tau)}(t)$、噪声物体颜色 $\textbf{c}_i^{(\tau)}$，还多加了一个**方差 $\widetilde\beta_i(t)$（表示每个采样点的不确定度）**，如下图式 $(11)$；同时保证所有采样点的方差预测值为正数，且都加上一个minimum importance（transient网络中学习的是噪声物体，所以都将预测值加上一点不确定度）得到 $\beta_i(t)$，如下图式 $(12)$。

![[Pasted image 20230809105622.png]]

最后同体渲染得到像素点的 $\textbf{C}_i(\textbf{r})$ 一样，也可以通过类似体渲染的方程得到像素点的方差 / 不确定度 $\beta_i(\textbf{r})$ 

![[Pasted image 20230809105920.png]]

对于损失函数的计算如下图：**第一项**目的是尽量减少像素颜色的差异，同时对于方差 / 不确定度较大的地方像素的差异值可以适当放松；**第二项**的目的是为了防止不确定性过大（例如趋于 $\infty$）导致第一项为0，即像素颜色差异任意，所以加了第二项防止不确定性过大；**第三项**式 $L_1$ 正则化是为了防止模型用 $\sigma_i^{(\tau)}$ 来解释静态物体，也就是防止transient网络部分学走了static的density

![[Pasted image 20230809111618.png]]

最后可以理解为什么 transient网络中后面部分不用 view direction 作为输入：对于static场景中的辐射信息可以通过 static 网络中的 $l^{(a)}$ 学习到，而transient网络主要学习的是瞬态物体的信息 / 噪声信息，包括体密度 $\sigma^{(\tau)}$ 和 不确定度/方差 $\beta$，对于**不确定处的地方color已经不是那么重要**了（因为在测试时是需要除去的），而且为了**防止学走每张图片的静态辐射信息**

## 3. 结果可视化

可以看到 static 网络输出的图像就是原本的静态物体图像；transient 网络输出的结果构成的图像（只用 $\sigma^{(\tau)}$ 和 $c^{(\tau)}$ 渲染）只包含瞬态障碍，且其比较模糊，符合上述的不确定性中对瞬态像素不用太过准确描绘；最后用的是前两部分网络合成的 Composite 图像来和 GT 进行带有不确定性（最右边图）进行损失计算

![[Pasted image 20230809115238.png]]

# 参考链接

[[1707.05776] Optimizing the Latent Space of Generative Networks (arxiv.org)](https://arxiv.org/abs/1707.05776)
[深度学习中的不确定性 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/98756147)

[[1703.04977] What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? (arxiv.org)](https://arxiv.org/abs/1703.04977)