# NeILF++：Inter-Reflectable Light Fields

![[Pasted image 20230818173052.png]]

## 1. motivation

将[[VolSDF-2021]]中获得的几何信息添加到[[NeILF-2022]]中来使用，同时保留了NeRF中原本的radiance field作为约束来使得in-radiance学习得更好（其他做法都扔掉，认为直接预测每个点的out-radiance没有用）

## 2. Procedure

整个训练过程分为3个stage：几何学习、材质和in&out-radiance学习、共同学习

![[Pasted image 20230818180129.png]]

上图横轴为三个时期，纵轴√为需要计算的损失，×为不需要计算的损失，$\downarrow$ 为减少损失占比

### 1. SDF Initialization

> 这部分主要来自于 [[VolSDF-2021]]

用 “SDF MLP" 来学习每个位置的SDF值，然后通过SDF来转化为体密度 $\sigma$，再根据体渲染方程：$$\begin{aligned}L_o^R&=\sum_{i=1}^N\ T_i\ (1-exp(-\sigma_i\delta_i))\ R(\textbf{x}_i,\pmb\omega_o)\\&where \quad T_i=exp(-\sum_{j=1}^{i-1}\sigma_j\delta_j)\end{aligned}$$学习每个像素的预测值，并于GT进行损失计算 $(L_{vol})$

### 2. Material Initialization

> 这部分主要来自于[[NeILF-2022]]

根据已有的几何信息学习 Material、in-radiance、out-radiance 三个网络

1. 根据alpha blending方法找到观察方向与场景的**交点位置 $\textbf{x}$ 和法线 $\textbf{n}$**
2. 从 "BRDF MLP" 中得到**BRDF值** $$B:\textbf{x}\rightarrow\{\textbf{b},r,m\}$$其中与Material有关的优化的一些损失计算 $L_{smooth},L_{lambertian}$参考[[NeILF-2022]]
3. 从交点位置 $\textbf{x}$ 往上半球方向采样 $\pmb\omega_i$，并查询 "Incident Light MLP" 得到 **in-radiance**：$$L:\{\textbf{x},\pmb\omega_i\}\rightarrow\textbf{L}$$
4. 将 BRDF参数、法线 和 in-radiance等已知值放到渲染方程：$$L_o^P(\pmb\omega_o,\textbf{x})=\int_{\Omega}f(\pmb\omega_o,\pmb\omega_i,\textbf{x})\ L_i(\pmb\omega_i,\textbf{x})\ (\pmb\omega_i\cdot\textbf{n})\ d\pmb\omega_i$$得到PBR color，然后与GT进行损失计算 $(L_{phys})$
5. 对于4步骤中的每个入射方向 $\pmb\omega_i$都进行back trace，得到back trace 的交点后将该交点带入 "Outgoing Radiance MLP" 得到 **out-radiance**，为了让 in-radiance 和 out-radiance 趋于一致（两个点之间的Inter-reflection，它们的radiance不应该改变），即计算损失 $(L_{ref})$：$$L_{ref}=||L(\textbf{x}_1,\pmb\omega_i)-R(\textbf{x}_2,\pmb\omega_i)||_1$$

### 3. Joint Optimization with Inter-reflections

最后同时执行 1和2 中的操作来**同时优化**，使得结果更准确

## 3. Contribution & Limitations

### 1. Conntribution

1. 结合了 VolSDF 和 NeILF 的优点，几何和材质方面都学习得很好，同时对**间接光照**也能充分学习
2. 利用了NeRF原本的 "Outgoing Radiance MLP" 来与 "Incident Light MLP" 互相约束，更准确快速学习

### 2. Limitations

1. 需要**用到[HDR](https://zhuanlan.zhihu.com/p/378840979)作为输入**（能保留更原本的光照、材质信息）
2. 每张照片**光照环境固定**
3. 目前位置包括这个都是主要关注opaque objects不透明物体，对于**半透明物体较难处理**
4. 虽然能得到BRDF和法线，但是隐式的 "Incident Light MLP" **对于relighting或者其他编辑工作没有帮助**，是否可以更好地拆开成更本质的东西（如visibility、光源信息）