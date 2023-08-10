# Related Work

## 1. anti-aliazing 方法

1. **super-sampling 超采样**：对每个像素cast multiple rays来得到像素值，但是这样会导致运行时间随着超采样率线性增长
2. **prefiltering**：通过发出视锥体而不是光线来与场景内容相交，然后通过查询符合视锥体相交面积的precomputed预计算好的multiscale多种大小的场景表示来得到像素点的值
	1. 但是我们现在不能得到场景的multiscale representations，因为我们不能提前得到场景的几何信息（我们是在恢复场景模型，不是传统的渲染场景），所以需要在**训练中学习 prefiltered representaions** 
	2. 我们需要的sacle是连续的，不是离散的（像mipmap中就是多个离散的纹理图），需要**学习一个可以查询任意scales的mip-NeRF神经网络**

## 2. 新视角合成的场景表示方法

1. **light field interpolation techniques光场插值技术**：如果拥有很多关于场景的images，就可以直接插值得到新视角而不用重建一个中间的表示。但是需要大量照片，还有其他问题
2. **explicit representations 显示表示场景**：例如1）Mesh-based represectations 基于网格的表示存储效率高、与现有图形学渲染管线较匹配。但是需要用到基于梯度下降方法来优化mesh geometry几何 和 topology结构，这种方法不连续且有局部最小值导致很难实现。2）直接用voxel grids体素格子来表示，或者使用梯度下降学习取训练深度网络来预测体素格子来表示场景。离散的voxel-based representations很有效，但是不能很好地在高分辨率场景下scale缩放
3. **coordinate-based neural representations基于坐标的神经表示**：将MLP作为一个隐式的连续函数，来将3D坐标映射到场景在该点的属性来表示3D场景

# Method

## 1. motivation 动机

### 1. NeRF出现的问题

1. 在train images都是同一full resolution的时候，测试的结果在于训练集相同的full resolution时表现好（左上角图），但是测试于训练集不同的 1/8 resolution时表现很差，有较多aliasing（左下角图）
2. 在train images是 multi-resolution时，测试full resolution时结果表现一般（模糊），且测试 1/8 resolution时表现也较差（锯齿）

![[Pasted image 20230809180116.png]] 

### 2. 原因

NeRF其对于每个像素点都**只发出一条光线** $\textbf{r}(t)=\textbf{o}+t\textbf{d}$（像素点中间位置）来在光线中取点采样。对于同一个位置的点，不同方向、分辨率的采样得到的特征都不一样（shape and size），但是NeRF中用点来采样就**忽略了这些特征**，把它们都看作是一样的，产生了相同的模糊的点的特征（如下图中蓝色和橙色的scale / resolution不一样，每个采样位置的信息应该与shape和size有关，即锥台包括的部分，但是NeRF只是当作一个相同的点），所以mult-resolutioin下渲染效果不好

![[Pasted image 20230809162913.png|400]] ^a0d95f

相同尺寸的一张图，像素点越多，分辨率越高（单位面积内像素点多）。所以对于高分辨率的图，其每个像素点的大小都比较小（更加精细），也就对应着上图中的蓝色部分，即锥体较瘦，相反黄色部分较肥，表示像素点更大，分辨率更低

### 3. 解决办法

1. **super-sampling 超采样**：对每个像素发出多条光线来进行超采样，提高采样率来anti-aliasing，但是这样耗时太多（与采样光线数成正比）
2. **Conical frustums 圆锥截台**：不再是发生一条光线来在光线上采样点，而是发出一个锥形，取每一部分**锥台来作为采样点**（这样不同resolution / scale在同一位置的不同信息 / 特征都能够通过锥台来得到）

这里的conical frustums相当于**通过 low-passed filter减少需要的 Nyquist 频率**，这样所需的采样点的数量就减少了

## 2. Cone Tracing and Integrated Positional Encoding

![[Pasted image 20230809205202.png]] ^6496e2

### 1. Cone Tracing

Mip-NeRF对每个像素点**发出一个cone圆锥体**，对圆锥体进行**采样得到多个 conical frustums 圆锥截台**。

[[#^a0d95f|上图中]]虚线所围的锥形是Mip-NeRF的采样锥台。原本的同一个采样点可能会因为不同的resolution / scale但是预测同一个值导致模型表现不好，现在对同一个位置不同方向的锥台不同（大小和形状不同）而会预测不同的值解决了这个问题

### 2. Integrated Positional Encoding

原本NeRF中是对采样点进行 PE 编码：$\gamma(\textbf{x})$ ；现在没有采样点了，采样的是一整个锥台，最简单及有效的方法就是求整个锥台的**所有经过PE编码后的点的期望值**

![[Pasted image 20230809204302.png]]

这里的 $F(\textbf{x},\textbf{o},\textbf{d},\dot{r},t_0,t_1)$ 表示的是点 $\textbf{x}$ 在是否在 $(t_0,t_1)$ 范围内；$\gamma^*(\textbf{o},\textbf{d},\dot{r},t_0,t_1)$ 表示的是在 $(t_0,t_1)$ 范围内的这个锥台的所有点编码值的期望（也就是要输入到MLP网络中的input）

但是这个式子没法有效地求出来，于是就用**多元高斯分布 $F(\textbf{x},\cdot)$ 来近似锥台内 $\textbf{x}$ 的分布**。称经过**多元高斯分布的positional encoding 的期望值为 Integrated Positional Encoding（IPE）**

#### 1. 求 $\textbf{x}$ 的多元高斯分布 

我们需要算出该分布 $F(\textbf{x},\cdot)$ 的**mean均值**和**covariance协方差**。因为锥台每个截面都是圆形，且锥台沿着中心轴对称，所以可以用**三个变量来表示该高斯分布**：沿着光线方向的均值 $\mu_t$，沿着光线方向的方差 $\sigma_t^2$，垂直于光线方向（半径方向，都是对称的）的方差 $\sigma_r^2$：

![[Pasted image 20230810104153.png]]

其中 $t_{\mu} = \frac{(t_0+t_1)}{2}$ 为沿着光线的中点，$t_{\delta}=\frac{(t_1-t_0)}{2}$为沿着光线方向的锥台半距离，上面三个变量的推导在最后

将 $\textbf{x}$ 的多元高斯分布从锥台坐标系中**转到世界坐标系**：

![[Pasted image 20230810104855.png]]

到现在我们成功表示了点 $\textbf{x}$ 在多元高斯分布下的表示，但是我们要求的 IPE 是经过编码后的 $\gamma(\textbf{x})$ 在多元高斯分布下的期望值，而从 $\textbf{x}$ 到 $\gamma(\textbf{x})$ 可以通过多元高斯分布的一些性质来进行类似的变化得到 $\gamma(\textbf{x})$ 的多元高斯分布：

#### 2. 求 $\textbf{Px}$ 的多元高斯分布

首先将 PE 变化中的三角函数内的部分 $\textbf{Px}$ 的 $\textbf{P}$ 写成矩阵形式：

![[Pasted image 20230810110135.png]]

由多元高斯分布的性质：若 $X\sim N(\mu,\Sigma)$，$Y=AX+B$，其中矩阵A、向量B满足 rankA = n，那么Y也服从n维多元高斯分布 $Y\sim N(A\mu+B,A\Sigma A^T)$。所以可以得到 $\textbf{Px}$ 的均值 $\pmb{\mu}_{\gamma}$ 和 协方差矩阵 $\pmb{\Sigma}_{\gamma}$：

![[Pasted image 20230810111302.png]]

#### 3. 求 $y(\textbf{Px})$ 的多元高斯分布的期望

对于服从高斯分布的变量 x，其对应的 sin(x) 和 cos(x) 的期望值如下：

![[Pasted image 20230810111613.png]]

我们将其转为求多元高斯分布的变量 $\textbf{x}$（这里的 $\textbf{x}$ 指的是 $\textbf{Px}$）的 $sin(\textbf{x})$ 和 $cos(\textbf{x})$，并带入到求 经过PE编码的$\gamma(\textbf{x})$的多元高斯分布的期望：

![[Pasted image 20230810112024.png]]

其中 $\circ$ 为矩阵的对应元素相乘，$diag(\pmb\Sigma_{\gamma})$ 为 $\pmb\Sigma_{\gamma}$ 的对角线元素（因为PE中的$3L$ 个维度/行中每一个维度都是独立的，所以只需要求边缘分布，也就是只需要对角线上的元素），也就是从原本的 $(3L,\ 3L)$ 维变成现在的 $(3L,\ 1)$ 维；而 $\pmb\mu_\gamma$ 为 $(3L,\ 3)$ 维；所以最终的 $\gamma(\pmb\mu,\pmb\Sigma)$ 为 $(2*3L,\ 3)$ 维（NeRF中的PE编码后的维度为 $(2*L,\ 3)$ ）

#### 4. 式子的优化

但是由于 $\pmb\Sigma_\gamma$ 为 $(3L,3L)$ 维，计算量过大且我们只需要对角线元素，所以可以改写成：

![[Pasted image 20230810114905.png]]

这样就不用求 $\pmb\Sigma_\gamma = \textbf{P}\pmb\Sigma\textbf{P}^T$ 了，而是用式 $(15)$ 来得到，其中 $\pmb\Sigma$ 为 $\textbf{x}$ 的多元高斯分布的协方差矩阵。

继续往前化简 $diag(\pmb\Sigma)$ 可以得到：

![[Pasted image 20230810115139.png]]

此式可以代替式 $(8)$ 中 $\pmb\Sigma$ 的计算

#### 从频率角度分析PE和IPE

Encodings里面分为很多行，每一行表示一个cos / sin的函数（值从-1到1随x变化），从最上面到最下面函数的频率逐渐减低（最上面值变化快，最下面变化慢）。Encoded Samples中表示的是当前采样点x的 $\gamma(x)$ 的值（不同颜色的格子表示对应频率下的 $cos/sin(2^i\textbf{x})$，中间图越上面 i 越大，也就是频率越大）

对于NeRF，那些高频特征由于超过采样频率，导致了渲染出现混叠，而产生aliasing；而Mip-NeRF的采样是一个锥台高斯分布的期望，是一段x的积分，对于**高频部分会包含多个周期而消除收缩到0**（Encodings图中高频部分会包括很多个周期，积分起来接近0），于是可以自动使高频部分向零收缩，避免混叠现象

![[Pasted image 20230809210623.png]]

## 3. Supplement 补充

### 1. 超参数 L

IPE每次采样**都能保证高频部分被弱化收缩于0**（因为每个锥台的高斯分布都是随着采样间隔变化的），所以就算输入 $\textbf{x}$ 经过编码后有很高的频率（L很大），高频部分由于收缩到0从而不会造成影响（不会因为采样率低于高频频率而出现aliasing），所以IPE中的L可以任意大

PE由于采样的是一个点，依然会保留采样到的高频信息，若L过大，会导致高频部分的频率大于采样频率，产生aliasing，所以PE中的L大小有限制

![[Pasted image 20230810091849.png|400]]

### 2. 将coarse和fine合并成一个网络

NeRF中用了两个网络coarse和fine，并使用采样点来采样，其**PE的特性决定它只能学习one single scale单一尺度的场景模型**，所以需要先经过均匀采样来放到coarse中学习场景大体信息（主要为低频），再根据权重重采样放到fine网络中学习场景更细致的信息（低频和高频），而**不能将两次采样结果都放到一个网络里面**（均匀采样和根据权重再次采样也类似于多规模场景，会出现[[#^a0d95f|上图问题]]）

但是Mip-NeRF采用conical frustums采样方法，其**IPE的特性决定它可以学习multi-scale 多尺度场景信息**，可以根据不同的scale/resolution**动态调整多元高斯分布的范围**来学习不同采样范围的高、低维信息，所以**只用一个网络**即可（为了效率也用了和NeRF一样的均匀粗粒度采样+根据权重细粒度采样，但是两次采样都放到同一个网络中输入训练，不过loss的权重不同而已）


前面部分为粗采样的结果的loss，后面部分维细采样的结果的loss，其中粗采样的loss还需要加入超参数 $\lambda=0.1$ 来调整比例

![[Pasted image 20230810102440.png]]

### 3. 锥台内点的多元高斯分布推导

![[9840338e43a9ebb9f016f14149e081e.jpg|300]] ![[9f3d345852498220bb9eb7ed7b282ed.jpg|300]]

下面接着推导 $\mu_t$、$\sigma_t$、$\sigma_r$

![[Pasted image 20230810170719.png|300]] ![[Pasted image 20230810170737.png|300]]

但是我直接推导 $\sigma_r^2$ 结果不对
![[21e7bac4cc1a160aa88d7283153cf4f.jpg|300]]

最后由于 $t_1$ 和 $t_0$ 的相差很小，而且经过高次幂作差再相除，得到的结果不精确（可能出现0或者NAN），所以改写成 $t_\mu=\frac{t_1+t_0}{2}$ 和 $t_\delta=\frac{t_1-t_0}{2}$

![[Pasted image 20230810171200.png]]