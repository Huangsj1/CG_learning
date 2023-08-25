# Learning Object-Compositional Neural Radiance Field for Editable Scene Rendering

> 通过将**背景和物体分隔开来**，形成 two-pathway 的两路结构。其中每个物体都能学到独有的特征

![[Pasted image 20230822220905.png]]

## 1. 网络结构

![[Pasted image 20230822220925.png]]

**已知内容**：需要知道**不同相机视角的图片** 和 **所有物体的2D masks**

网络主要包括两个分支：**Scene branch** 和 **Object branch**。前者主要学习场景的内容；后者主要学习每一个物体的内容

输入部分多了体素特征（具体要参考[NSVF](https://zhuanlan.zhihu.com/p/444499451)），其中$f_{scn}$作为两个分支的输入，帮助**场景和物体分支学习物体遮挡的区域**；$f_{obj}$为物体分支独有，学习的是**物体的decomposition分解内容**（应该指的是所有物体共有的一些信息？）。其中$f_{scn},f_{obj}$是由该点$\textbf{x}$所在的block的8个顶点（立方体的八个顶点）的值插值得到的，可以利用**局部小空间的属性信息**（几何、材料、颜色等），简化后续radiance field的学习

每个物体都有一个对应的**可学习的 Object Activation Code**，用来学习物体独有的特征

## 2. Training

### 1. 场景的训练

直接对所有像素发射光线采样输入到scene branch中得到预测图像，与GT进行loss

![[Pasted image 20230822223501.png]]

### 2. 物体的训练

每个物体都要输入对应的 object activation code到object branch中；而且需要得到**所有光线对应的像素值**。下面是一个obj物体对每根光线采样N个点的体渲染方程计算

![[Pasted image 20230822223849.png|500]]

**完整的过程**：

1. 对**每个像素**发射一条光线 $\textbf{r}\in N_r$
2. 每根光线 r 都需要对**所有 K 个物体进行object branch**的计算得到 颜色 $\widehat{C}(\textbf{r})^k_{obj}$ 和 不透明度 $\widehat{O}(\textbf{r})^k_{obj}$
3. 损失函数的计算包括两部分：最小化**渲染图像与GT的颜色误差** 和 最小化**渲染图像与GT的mask的体密度误差**
	1. 颜色误差：只计算当前物体mask内的误差（外边部分与当前物体无关就不用管）
	2. 体密度误差：在当前物体的mask内就需要让不透明度接近1，否则就接近0
	* $M(\textbf{r})^k$ 表示当前光线 r 是否与当前物体 k 的mask相交（即像素点是否在mask内），若相交则为1，否则为0
	* $w(\textbf{r})^k$ 来平衡不透明度（感觉没什么用啊，都有 $\lambda_2$来平衡两种误差的权重）

![[Pasted image 20230822225723.png]]

### 3. 场景+物体的损失计算

$$L=L_{scn}+L_{obj}$$

## 3. 采样的优化

![[Pasted image 20230822210515.png]]

他这里是对所有的光线下所有的物体都采样，然后在mask内的光线会计算颜色的损失，且让不透明度往1的方向学习，不在mask内的光线颜色损失为0，且让不透明度往0的方向学习。但是**不在物体的mask内的部分可能是occluded被遮挡或者不存在**，这种歧义就可能导致切掉物体的部分内容且学到一个破碎的radiance field；同时不能够直接将不在mask内的光线给删掉，因为**不在mask的内容也可以学到与物体有关的信息**（比如物体在其他部分没有内容），删掉了可能导致渲染的时候在那些不在mask的部分出现奇怪的东西（因为这部分没有被学些到）。所以就通过下面的方法来使得采样更加准确

1. **Scene Guidance**：根据scene branch中得到的透明度transmittance来指导object branch 采样，从而减少在被遮挡区域的采样点（具体怎么个指导法？）
2. **3D guard mask**：原本对于每个物体都会对所有光线采样，现在通过该方法只对在物体的mask内的光线采样
	1. 从scene branch中得到场景的深度 $d_{scn}$（上左图灰色边界线），并将深度往前延伸 $\epsilon$
	2. 然后将光线**非mask的部分的深度在 $d_{scn}+\epsilon$ 之后的作为3D guard mask**
	3. 训练的时候直接**删去3D guard mask中的采样点**

## 4. Contributions & Limitations

### 1. Contributions

1. 将场景单独作为一部分，所有物体作为另一部分，其中每个物体又都用一个可学习的object activation code来学习独有的特征
2. 损失函数的除了颜色计算外，还包括物体的**不透明度计算**，让mask部分和非mask部分分别往1和0去学习
3. 优化采样，**不在非mask中的阻塞和其他物体中采样**，防止梯度反传到其他相关区域参数上；但是**保留了非mask的其他部分采样**

### 2. Limitations

1. 物体编辑没讲清楚如何做的，是重新学习物体的object activation code还是通过变换得到
2. 对编辑后的物体的**阴影**部分没处理好，看上去很假

# 疑惑

1. 体素那一块的特征学的是什么❓（估计要先看[NSVF](https://zhuanlan.zhihu.com/p/444499451)）
2. 他这里讲编辑的时候scene branch对于非目标区域不变，目标区域不采样；而根据user-defined manipulation来将物体的颜色和不透明度转换到需要的地方，但这具体是如何进行编辑的❓