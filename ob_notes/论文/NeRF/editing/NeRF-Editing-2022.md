# NeRF-Editing：Geometry Editing of Neural Radiance Fields

> 将NeRF得到的implicit模型转换成**explicit的网格模型**，然后对**网格模型进行编辑**，根据编辑前后来**修改体渲染中的点的位置**（即将当前/编辑后的点映射回原来/编辑前的点），最后得到编辑后的渲染图像

![[Pasted image 20230824094336.png]]

## 1. 编辑与渲染

![[Pasted image 20230824094444.png]]

### 1. 得到预训练好的网络

可以用**预训练好的NeRF** $F_\Theta:(\gamma(\textbf{p}),\gamma(\textbf{d})))\rightarrow(\sigma,\textbf{c})$ 来表示该3D场景；但是在网格提取的过程中NeRF表现得一般，所以这里还是**采用了[NeuS](https://arxiv.org/abs/2106.10689)方法来用SDF表示几何特征**

### 2. 网格的提取与场景的编辑

![[Pasted image 20230824101834.png|400]]

1. **得到三角网格**：使用[Marching cubes](https://zhuanlan.zhihu.com/p/561731427) 算法得到**场景的三角网格表示 Extracted Triangular Mesh $S$**（根据距离SDF不断缩小平面来得到表面）
2. **用户编辑三角网格**：使用 [ARAP](https://zhuanlan.zhihu.com/p/25846219)（as-rigid-as-possible）变形方法来让用户交互编辑网格得到 Deformed Triangular Mesh $S'$（通过移动控制点来编辑）。其中原本的三角网格中的顶点为 $\textbf{v}_i$，经过变形后顶点为 $\textbf{v}_i'$（使用其他变形方法也可以）
	* ARAP在保持局部细节的情况下编辑大体形态，保持每个刚性变形单元（就是一个顶点与周围顶点构成的部分）的**形状不变**，仅发生**旋转变换**（点与点之间的距离不变）。做法是最小化ARAP energy，即保持整个网格的稳定性和最小化所有的distortion energies变形能量（也就是只希望所有边只发生刚性形变）：$$E(S')=\sum_{i=1}^n\sum_{j\in N(i)}w_{ij}||(\textbf{v}_i'-\textbf{v}_j')-\textbf{R}_i(\textbf{v}_i-\textbf{v}_j)||^2$$这里的 $N(i)$ 表示顶点 i 的下标；$w_{ij}=\frac{1}{2}(cot\alpha_{ij}+cot\beta_{ij})$ 表示cotangent权重；$\textbf{R}_i$ 表示顶点 $i$ 到顶点 $i'$ 的最优旋转矩阵
3. **下面用Tetrahedral四面体网格取代Triangular三角网格来与Continuous Volume Space建立一致性的原因**：若直接从三角网格来进行渲染会产生很多artifacts，如下图无论是找Closest Point最近点还是3NN插值三个最近点得到当前点效果都不够好，都会导致不连续的现象（可能是每个三角只有一个面，采样多个点的时候只有很少的采样点能够得到表面信息，而四面体能够更好地得到表面几何信息）![[Pasted image 20230824111142.png|500]]
4. **Tetrahedral Mesh四面体网格的获取**：
	1. **先获取Bounding Cage Mesh**：通过扩大三角形网格，并且将每个三角形往外移动一点距离就能够得到包裹住整个三角网格的几何体，相当于得到场景的外壳一样的东西（但是这里具体用了什么方法没讲）
	2. **生成Tetrahedral Mesh**：使用 [TetWild](https://arxiv.org/abs/1908.03581)方法从cage mesh中得到四面体网格 $T$（就是用很多个四面体来建模出物体表面）
5. **根据三角网格来编辑四面体网格**：使用三角网格顶点 $\textbf{v}_i$ 来驱动四面体网格 $T$ 的变形得到 $T'$，其中四面体网格中的点也从 $\textbf{t}_k$ 变成 $\textbf{t}_k'$。具体算法也可用 ARPR 来做，但是需要转变一下，将三角网格中的顶点用重心坐标在四面体的四个顶点中表示，即 $\textbf{A}\textbf{t}'=\textbf{v}'$，其中 $\textbf{A}$ 为顶点 $\textbf{v}'$ 所在四面体的重心坐标，优化方程为：$$min\ E(T'),\quad subject\ to\ \textbf{A}\textbf{t}'=\textbf{v}'$$

### 3. Ray Bending 光线弯曲

![[Pasted image 20230824114643.png|200]]

在有了变形前后的几何显示描述后，我们可以直接对体渲染中**当前/编辑后采样点的位置进行映射回到原本/编辑前采样点的位置**，这样就达到了编辑的目的（有点像[[ST-NeRF-2021]]中得到修改bounding box的仿射变换后对点坐标进行逆仿射变换）

**具体做法**：同NeRF中对光线采样得到采样点，得到采样点所在的**编辑后的四面体**，这同时也能得到编辑前的该四面体的位置，计算该点在**编辑前的四面体中的位置**，计算得到编辑后的与编辑前的**位置偏移 $\Delta p$**，于是就能得到当前点在编辑前所在的位置 $\Delta p$，传入NeRF网络中：$$(\zeta(\textbf{p}+\Delta\textbf{p}),\ \zeta(\textbf{d}))\rightarrow(\sigma,\textbf{c})$$对于不在四面体网格中的点就不用移动/映射回去

## 2. Contributions & Limitations

### 1. Contributions

1. 从**显示表示**出发来编辑场景，**不用重新训练网络**，只需要根据变形前后得到编辑前点的位置再带入网络即可

### 2. Limitations

1. 没有将**光照和阴影**考虑进去，对于阴影部分编辑后还是同样的颜色
2. **过程复杂耗时**，需要先得到显示网格表示，然后对网格编辑，再经过NeRF重新渲染
3. 只能对场景的某个部分进行形状编辑（应该也可以移动），**不能够进行material材质、色彩编辑**