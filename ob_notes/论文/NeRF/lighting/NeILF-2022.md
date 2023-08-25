# NeILF：Neural Incident Light Field

![[Pasted image 20230818123126.png]]

## 1. Rendering Equation

这里不用像NeRF一样的体渲染方程，而是取**物体表面的点 $\textbf{x}$**（需要object mesh来作为输入，从而得到物体表面的点）来用渲染方程：$$L_o(\pmb\omega_o,\textbf{x})=\int_\Omega f(\pmb\omega_o,\pmb\omega_i,\textbf{x})\ L_i(\pmb\omega_i,\textbf{x})\ (\pmb\omega_i\cdot\textbf{n})\ d\pmb\omega_i\tag{1}$$其中BRDF项用 Disney BRDF来表示：$$\begin{aligned}f(\pmb\omega_o,\pmb\omega_i,\textbf{x})&=f_d+f_s(\pmb\omega_o,\pmb\omega_i) \\ &=\frac{1-m}{\pi}\cdot\textbf{b}\ +\ \frac{D(\textbf{h};r)\cdot F(\pmb\omega_o,\textbf{h};\textbf{b},m)\cdot G(\pmb\omega_i,\pmb\omega_o,\textbf{h};r)}{(\textbf{n}\cdot\pmb\omega_i)\cdot(\textbf{n}\cdot\pmb\omega_o)} \end{aligned}\tag{2}$$$\textbf{h}$ 表示半程向量，与BRDF有关的：3维base color $\textbf{b}$、1维金属度metalness $m$、1维粗糙程度$r$

## 2. Motivation & Contribution

![[Pasted image 20230818115535.png]]

### 1. Neural Incident Light Field MLP

NRF没有间接光照且需要光源和相机同一位置，NeRV虽然有间接光照但需要光源已知，NeRD虽然用SG模拟光源但是光照高频恢复不好且无法处理间接光照，Neural PIL虽然用环境贴图模拟光源但是也无法处理间接光照，NeRFactor的环境贴图无法处理间接光照。

于是这里直接用一个**神经网络MLP来学习每个点不同方向的入射光照**值，即式$(1)$中的 $L_i(\textbf{x},\pmb\omega_i)$：$$L:\{\textbf{x},\pmb\omega\}\rightarrow \textbf{L}\tag{3}$$

用蒙特卡洛积分对 $\pmb\omega_i$ 进行采样，因为是在半球上，所以 $pdf(\pmb\omega_i)=\frac{1}{2\pi}$，总共采样 $S_L$个点，于是得到离散的方程 $(1)$ 的表示：$$L_o(\pmb\omega_o,\textbf{x})=\frac{2\pi}{S_L}\sum_{i\in S_L}f(\pmb\omega_o,\pmb\omega_i,\textbf{x})\ L_i(\pmb\omega_i,\textbf{x})\ (\pmb\omega_i\cdot\textbf{n})\tag{4}$$

### 2. Disney BRDF MLP

与BRDF有关的参数：3维base color $\textbf{b}$、1维金属度metalness $m$、1维粗糙程度$r$ 也直接用一个神经网络MLP来学习：$$B:\textbf{x}\rightarrow\{\textbf{b},r,m\}\tag{5}$$

## 3. Rendering

对于物体表面的点 $\textbf{x}$（可以理解为采样点的期望点）和观察方向 $\pmb\omega_o$，可以通过体密度取负梯度得到法线 $\textbf{n}$；可以通过采样得到入射方向 $\pmb\omega_i$ ；通过两个MLP可以得到BRDF有关的值而算出 $f$ 和得到 $L_i$，这样就能算出渲染方程 $L_o$ 的值得到像素点color

## 4. Loss

上面的两个MLP同时学习 Lighting 和 BRDF，可能会出现ambiguity（即BRDF学得很差，但是Lighting足够复杂就可以展现出和GT一样的效果），但是在[[NeRF++-2020]]中也解释了MLP不擅长学习太过复杂，变化太大（高频）的东西，所以会让 Lighting 和 BRDF 都更加平滑，也就能各自学得更加准确。但是为了更好显示加快学习和增加robust，下面增加了两中损失函数来优化

### 1. Bilateral Smoothness 双向平滑

鼓励粗糙度 $r$ 和 金属度 $m$ 都更加平滑：$$l_{smppth}=\frac{1}{S_I}\sum_{\textbf{p}\in S_I}(||\nabla_{\textbf{x}}r(\textbf{x}_{\textbf{p}})||+||\nabla_{\textbf{x}}m(\textbf{x}_{\textbf{p}})||)e^{-||\nabla_{\textbf{p}}\textbf{I}(\textbf{p})||}\tag{6}$$让 $r$ 和 $m$ 的偏导都更小/平滑，$S_I$ 为 所有的像素，$\nabla_{\textbf{p}}\textbf{I}(\textbf{p})$ 为输入图像每个像素的偏导（可以预计算）

### 2. Lambertian Assumption 漫发射假设

这里还假设物体表面趋向于Lambertian，有着更高的粗糙度（$r$ 更接近1），更低的金属度（$m$ 更接近0）：$$l_{lambertian}=\frac{1}{S_I}\sum_{\textbf{p}\in S_I}(|r(\textbf{x}_{\textbf{p}})-1|+|m(\textbf{p})|)\tag{7}$$

### 3. Rendering Loss

损失还需包括最终渲染出来的图像与GT的损失：$$l_{image}=\frac{1}{S_I}\sum_{\textbf{p}\in S_I}||\textbf{I}_{\textbf{p}}-L_o(\textbf{x}_{\textbf{p}},\pmb\omega_o)||_1\tag{8}$$

### 4. Final Loss

最终所有损失加起来：$$l=l_{image}+w_sl_{smooth}+w_ll_{lambertian}\tag{9}$$其中 $w_s=10^{-4}$ 和 $w_l=10^{-3}$ 为系数

## 5. Contribution & Limitations

### 1. Contribution

用**MLP来学习渲染方程中的 $L_i$**，即学习物体交点的接收到各个方向的radiance，而不是学习Visibility / 光源Light，能够得到inter-reflection**物体间反射**

### 2. Limitations

1. 需要**场景的mesh作为输入**，也就是需要准确的场景模型表示来得到观察方向与场景**交点**（这里可以直接用NeRF来建模并得到期望交点作为交点，也可以用其他如VolSDF来建模场景，这也是NeILF++所做的）
2. 使用**relighting就需要重新完整地光线追踪**，即NeILF对新的光源产生的影响的计算没有任何作用