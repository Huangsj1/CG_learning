# NeRF-In：Free-Form NeRF Inpainting with RGB-D Priors

>在多视角合成中，通过用户给定的一个视角的**需要删除的物体的mask**，**生成多个视角的mask**，然后借助已有方法得到这些视角的**删除物体后的图像和深度图**，将其与最初NeRF网络得到的图像和深度图计算损失来**更新NeRF**学到删除物体后的图像

![[Pasted image 20230823163319.png]]

## 1. 网络结构

网络**只采用NeRF的网络结构**，根据用户给定的mask来得到多个视角删除mask中的物体后的图像，将其作为GT**直接更新网络**

## 2. Inpainting 修补的具体步骤

![[Pasted image 20230823163420.png]]

1. 先得到**预训练的NeRF模型**：$$I=F^{image}_{\Theta}(\textbf{o})$$
2. **用户输入**：用户需要给定的相机视角 $\textbf{o}_u$ 和该视角下需要**删除的物体的mask $M_u$**（mask内$M_u=0$，mask外$M_u=1$）；通过NeRF网络得到该视角下的图像$I_u$和深度图$D_u$
3. **采样多个视角**：根据[LLFF](https://arxiv.org/abs/1905.00889)中对三维物体视角采样的方法得到**K个视角下**的相机位置$\textbf{O}=\{\textbf{o}_s|s=1...K\}$ ，并通过网络得到对应的图像 $\textbf{I}=\{I_s|s=1...K\}$ 和深度图 $\textbf{D}=\{D_s|s=1...K\}$
	* 下面4，5，6的操作中同时也对用户的mask、图像和深度图进行操作得到对应的user-chosen image $I^G_u$ 和 depth image $D^G_u$
4. **多视角的masks获取**：根据已有的用户视角下的mask，使用 video object segmentation视频目标分割方法 [STCN](https://arxiv.org/abs/2106.05210)来得到**不同视角下的masks** $\textbf{M}=\{M_s|s=1...K\}$（这里应该是用多视角来取代video的不同帧吧？）
5. **多视角的guiding images获取**：使用[MST inpainting network](https://arxiv.org/abs/2103.15087)方法来**将带有mask $\textbf{M}$ 的原本的图像 $\textbf{I}$ 修复得到guiding images** $\textbf{I}^G=\{I_s^G|s=1...K\}$（该方法是将图像中有mask的部分当作未知并**还原**出来）：$$I^G_s=\rho(I_s,M_s)$$
6. **多视角的guiding depth images获取**：使用 [The Fast Bilateral Solver](https://arxiv.org/abs/1511.03296)快速双边求解器 来**将带有mask $\textbf{M}$ 的原本的深度图 $\textbf{D}$ 平滑处理得到guiding images** $\textbf{D}^G=\{D^G_s|s=1...K\}$（该方法给定原始图像RGB像素图 $\textbf{I}^G$、原始深度图 $\textbf{D}$、置信图/mask图 $\textbf{M}$，将置信图中正确率低，也就是mask图中的mask部分对应的深度图进行平滑处理**得到新的更平滑可靠的深度图** $\textbf{D}^G$）：$$D^G_s=\tau(D_s,M_s,I^G_s)$$
7. **损失函数计算**：损失函数包括颜色的损失 $L_{color}$ 和深度的损失 $L_{depth}$：$$L=L_{color}(\Theta)+L_{depth}(\Theta)$$
	1. **color 损失**：将所有的RGB图像分为用户给定的 $\textbf{O}^{all}=\textbf{o}_u$ 和其他视角下预测的 $\textbf{O}^{out}=\textbf{O}$。对于**用户给定的需要计算整个图像**的颜色损失；对于**其他视角预测的只需要计算非mask部分**的图像的颜色损失：$$\begin{aligned}L_{color}(\Theta)&=L^{all}_{color}(\Theta)\ +\ L^{out}_{color}(\Theta)\\&=(\sum_{\textbf{o}\in\textbf{O}^{all}}F^{image}_{\Theta}(\textbf{o})-I^G_o)\ +\ (\sum_{\textbf{o}\in\textbf{O}^{out}}(F^{image}_{\Theta}(\textbf{o})-I^G_o)\odot M_o)\end{aligned}$$paper中经过实验测试，越多的图像加入到 $\textbf{O}^{all}$ 中，图像导致更加多的**inconsistencies不一致问题**发生，所以只是用用户给定的准确的mask来学习mask内的颜色值
	2. **depth 损失**：如果只有color损失，会产生**错误的深度信息**，进而产生**错误的几何信息**，出现不应该存在的物体，所以还需要添加深度损失来得到准确得几何信息：$$L_{depth}(\Theta)=\sum_{\textbf{o}_s\in\textbf{O}}||D^f(\textbf{o}_s)-D^G(\textbf{o}_s)||^2_2\ +||\ D^c(\textbf{o}_s)-D^G(\textbf{o}_s)||^2_2$$

## 3. Contributions & Limitations

### 1. Contributions

1. 除了用RGB图像外还用了**深度图**来更新网络，得到更准确的几何信息
2. 提供了得到**得到不同视角的masks**方法、基于**图像的还原mask部分**的方法、基于**masks和图像来平滑处理深度图**的方法

### 2. Limitations

1. 利用已有的方法得到多个视角的除去mask部分的图像，通过这些图像重新训练NeRF，这相当于用新的图像来训练NeRF，不仅**效率低**，而且**效果也一般**，感觉没什么创新