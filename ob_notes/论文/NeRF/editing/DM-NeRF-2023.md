# DM-NeRF：3D Scene Geometry Decomposition And Manipulation From 2D Images

> 通过给定的**多个视角图片**以及对应的**所有像素点所属物体的标签**，来学习**NeRF网络** 以及 一个能得到**所有采样点所属物体标签的Object Field网络**；将用户对物体的平移、旋转、缩放操作用**矩阵**来表示，渲染编辑后的图像时**直接对采样点进行修正**来得到编辑后的图像

![[Pasted image 20230824172232.png]]

## 1. Training——Object Field Representation

> 在学习**NeRF网络**的同时学习整个**3D空间中点的object code网络**

![[Pasted image 20230824172426.png]]

### 1. Object Field Representation 物体标签场的表示

对于每个采样点 $\textbf{p}$，除了通过NeRF学习到 $(\sigma,\textbf{c})$ 之外，还学习所属物体的标签，其中使用 $H+1$ 维的one-hot表示物体标签（前H维表示物体，最后一维表示空气，因为场景中确实有非物体/空气部分，而且后面的编辑操作也需要考虑是否有物体，所以需要多加一维表示空气）：$$\textbf{o}=f(\textbf{p}),\quad where\ \textbf{o}\in\mathbb{R}^{H+1}$$

### 2. Object Code Projection 物体编码投影

将当前像素对应的光线的所有**采样点按权重**来得到像素的所属物体标签/编码（K表示K个采样点）：

![[Pasted image 20230824172848.png]]

### 3. Object Code Supervision 物体编码监督

为了生成正确的物体投影编码，这里**需要提供所有视角对应的物体编码图像**，即需要知道每个视角下所有像素所属的物体编码

**目标**：

1. 对于**object的点**使得2D的**物体预测图要与GT物体预测图相同**（这里只考虑物体的编号，即前H维）
2. 对于**empty的点**让最后一维（H+1维）预测为1，前H维预测为0；

#### 1. 2D object

![[Pasted image 20230824151358.png|300]]

运用 sIoU 和 CES 方法解决**预测的物体编号的维度$H$与GT的维度$T$不同**的问题，计算损失让得到的2D的物体分类预测图 $\{I_1...I_l...I_L\},I_l\in\mathbb{R}^{U*V*(H+1)}$ （其中UV分别表示宽高，(H+1)表示有H+1维的输出）与人工注释的2D GT分类图 $\{\overline{I}_1...\overline{I}_l...\overline{I}_L\},\overline{I}_l\in\mathbb{R}^{U*V*T}$ 进行损失计算 $L_{2d\_obj}$，使得预测得更加准确

![[Pasted image 20230824151925.png]]

#### 2. 3D empty & 3D object

![[Pasted image 20230824153332.png|300]]

上图紫色的点为表面预估点，绿色点为empty点，红色点为表面点附近的点，黑色的点可能为empty也可能为object的，所以就不将其用来监督empty部分；**表面距离d**的计算方法

![[Pasted image 20230824154305.png]]

对于每个采样点，都需要得到它的**surfaceness score表面得分 $s_k$ 和 emptiness score空白得分 $e_k$ 来判断它是在表面还是空白**的（越接近表面$s_k$越接近1，远离就接近0；$e_k$ 只用来判断在表面附近以及之前的采样点，因为后面的采样点不确定是否为空）

![[Pasted image 20230824154513.png]]

接着就可以计算损失 $l_{3d\_empty}$ 使得空白点的预测标签的第 $(H+1)$ 维 $o_k^{H+1}$ 预测为1（左边部分），物体点预测为0（右边部分）

![[Pasted image 20230824154829.png]]

同时计算损失 $l_{3d_obj}$ 使得空白点的预测标签的前 $H$ 维 $o_k^{h}$ 预测为0

![[Pasted image 20230824155216.png]]

#### 3. 最终损失

$$l=l_{2d\_obj}+(l_{3d\_empty}+l_{3d\_obj})$$


## 2. Manipulation——Inverse Quert Algorithm

![[Pasted image 20230824170748.png]]

* **Input**：
	1. 给定需要移动的物体的object code $\textbf{o}_t$
	2. 编辑操作 $\{\Delta\textbf{p},\textbf{R}^{3*3},t\geq0\}$（平移、旋转矩阵、缩放）
	3. 当前光线 $\textbf{r}$ 的所有采样点 $\{\textbf{p}_1...\textbf{p}_k...\textbf{p}_K\}$（下面的操作应该让**所有的光线一起执行**，否则可能导致物体重复移动）
* **Preliminary step**：
	1. 获得当前像素的projected object code $\hat{\textbf{o}}$
* **Main step**：
	1.  对所有的采样点进行操作 $\textbf{p}_k\ in\ \{\textbf{p}_1...\textbf{p}_k...\textbf{p}_K\}$：
		1. 计算当前采样点 $\textbf{p}_k$ 的 inverse point逆点 $\textbf{p}_{k'}$（编辑之前的点，即需要移动到当前采样点位置的点）：$$\textbf{p}_{k'}=\frac{1}{t}\textbf{R}^{-1}(\textbf{p}_k-\Delta\textbf{p})$$
		2. 获得当前点 $\textbf{p}_k$ 的 $\{\sigma_k,\textbf{c}_k,\textbf{o}_k\}$
		3. 获取逆点 $\textbf{p}_{k'}$ 的 $\{\sigma_{k'},\textbf{c}_{k'},\textbf{o}_{k'}\}$
		4. 处理视角阻塞情况：如果当前点为目标物体 且 被挡在后面，就将当前点object code改为projected object code表面物体标签（为什么需要这样处理？个人认为这里不需要操作）
		5. 获得当前采样点 $\textbf{p}_k$ 经过编辑后的各种值 $\{\bar{\sigma}_k,\bar{\textbf{c}}_k,\bar{\textbf{o}}_k\}$：
			1. 如果当前点不是目标物体 且 不是空气 且 逆点是需要移动到这的目标物体，那么发生了碰撞，需要退出（这里点很少的情况下应该不用直接退出，点多了才不能移动直接退出）
			2. 如果当前点不是目标物体 且 是空气 且 逆点需要移动到这，那么直接将逆点信息作为当前点的信息 $\{\bar{\sigma}_k,\bar{\textbf{c}}_k,\bar{\textbf{o}}_k\}\leftarrow\{\sigma_{k'},\textbf{c}_{k'},\textbf{o}_{k'}\}$
			3. 如果当前点是目标物体 且 逆点需要移动到这，那么直接将逆点信息作为当前点的信息 $\{\bar{\sigma}_k,\bar{\textbf{c}}_k,\bar{\textbf{o}}_k\}\leftarrow\{\sigma_{k'},\textbf{c}_{k'},\textbf{o}_{k'}\}$
			4. 如果当前点是目标物体 且 逆点不需要移动到这，那么这个点需要清零$\{\bar{\sigma}_k,\bar{\textbf{c}}_k,\bar{\textbf{o}}_k\}\leftarrow\{0,0,0\}$
			5. 如果当前点不是目标物体 且 逆点不需要移动到这，那么保留原来信息 $\{\bar{\sigma}_k,\bar{\textbf{c}}_k,\bar{\textbf{o}}_k\}\leftarrow\{\sigma_k,\textbf{c}_k,\textbf{o}_k\}$
			* 1，2，5为当前点不是目标物体的三种情况：逆点需要移过来但被阻塞，逆点需要移过来且可以移过来，逆点不需要移过来
			* 3，4为当前点是目标物体的两种情况：逆点需要移过来，逆点不需要移过来
	2. 得到所有采样点的信息 $\{\bar{\sigma}_k,\bar{\textbf{c}}_k,\bar{\textbf{o}}_k\}$ 后，计算像素的信息：颜色、物体标签

## 3. Contributions & Limitations

### 1. Contributions

1. 通过学习**场景所有点的所属物体标签**，这样就能够**单独地对某个物体编辑**，而且**编辑操作也用矩阵来表示**，不用重新训练网络（同[[ST-NeRF-2021]]和[[NeRF-Editing-2022]]一样通过矩阵映射和逆映射来编辑）
2. 考虑物体标签的同时也**考虑了空气类型**，使得编辑操作更加精准

### 2. Limitations

1. **没有考虑Lighting光照**，移动物体后光照对场景可能有很大影响，如阴影、亮度等方面
2. **只能对物体进行空间上的整体编辑**，即通过矩阵来平移、旋转、缩放，不能像[[ST-NeRF-2021]]和[[NeRF-Editing-2022]]一样编辑局部，也不能像其[[Edit NeRF-2021]]一样编辑材质颜色
3. 需要所有视角下的像素点所属物体标签