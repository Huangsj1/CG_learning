# VolSDF：Volume Rendering of Neural Implicit Surfaces

## 1. motivation

对于NeRF来说体渲染方程的积分没有问题，但是通过coarse-to-fine来采样过程中，由于**采样点的选择比较粗糙**，导致体密度 $\sigma$ 预测得不够准确，从而引入了误差，导致**几何形状出现noisy、artifacts**

## 2. Method

不再是直接通过直接一个MLP来学习预测 $\sigma$，而是转化为**用Signed Distance Function（SDF）这种几何的函数来表示 $\sigma$**，转而用**MLP来学习SDF函数**

这种方法有利于**学习形状的inductive bias归纳偏置**，同时还能够**限制 Opacity不透明度 的误差边界来限制误差**，具体做法是通过设计相对应的**采样**过程来减小这一误差

## 3. 各部分细节

### 1. SDF to Density

* $\Omega\subset\mathbb{R}^3$：在物体内部的点
* $M=\partial\Omega$：表面边界
* SDF表示**点到物体最近的表面的距离**（有符号）：$$d_\Omega(\textbf{x})=(-1)^{1_\Omega(\textbf{x})}\underset{\textbf{y}\in M}{min}||\textbf{x}-\textbf{y}||,\quad\quad and\ 1_\Omega(\textbf{x})=\left\{\begin{array}{l}1\quad if\ \textbf{x}\in\Omega\\0\quad if\ \textbf{x}\notin\Omega\end{array}\right.\tag{1}$$点 $\textbf{x}$ 在物体内部符号为负，在物体外部符号为正，距离越远绝对值越大
* SDF 转 $\sigma$ 的表示：$$\sigma(\textbf{x})=\alpha\ \Psi_\beta(-d_\Omega(\textbf{x})),\quad\quad and\ \Psi_\beta(s)=\left\{\begin{array}{l}\frac{1}{2}exp(\frac{s}{\beta})&if\ s\leq0\\1-\frac{1}{2}exp(-\frac{s}{\beta})&if\ s>0\end{array}\right.\tag{2}$$其中 **$\Psi_\beta$ 为Laplace distribution 的累积分布函数CDF**（均值为0，$\beta$为缩放程度，也是超参数）；这里的 $\alpha,\beta>0$ 的超参数（需要学习的） ![[Pasted image 20230819111307.png]]上图是一个Laplace 分布的例子，红色线就是 $\Psi_\beta(s)$。当 s < 0 时，点在物体外面，且越远值（这里可以看成是 $\sigma$）越小；当 s > 0 时，点在物体里面，且越远值越大；当 s = 0 时，$\sigma(\textbf{x})=\frac{\alpha}{2}$ ，点在物体表面

所以只要通过网络能得到点 $\textbf{x}$ 对应的SDF $d_\Omega(\textbf{x})$（也就是用MLP去学习$(1)$），就能直接直接带入$(2)$中得到体密度 $\sigma$

### 2. Volume Rendering Equation

沿着光线的透明度Transparency $T$ 计算如下：$$T(t)=exp(-\int_0^t\sigma(\textbf{x}(s))\ ds)\tag{3}$$那么不透明度Opacity $O$ 可以表示如下：$$O(t)=1-T(t)\tag{4}$$$O(t)\in[0,1]$ 且是单调递增的，所以可以将其看出是一个分布的CDF，对其求导可以得到原本的PDF：$$\tau(t)=\frac{dO}{dt}(t)=\sigma(\textbf{x}(t))\ T(t)\tag{5}$$$\tau(t)$ 表示的是当前点的**不透明度的PDF**，那么将其乘以当前点的 out-radiance $L$ 并进行积分就可以得到当前光线的color，也就是NeRF中的体渲染方程（$\textbf{c}$ 为相机原点，$\textbf{v}$ 为观察方向）：$$I(\textbf{c},\textbf{v})=\int_0^\infty L(\textbf{x}(t),\textbf{n}(t),\textbf{v})\ \tau(t)\ dt\tag{6}$$将其化为离散形式（采样点间隔为 $\delta_i$）可以得到：$$\begin{aligned}I(\textbf{c},\textbf{v})&=\int_0^M L(\textbf{x}(t),\textbf{n}(t),\textbf{v})\ \tau(t)\ dt\ +\ \int_M^\infty L(\textbf{x}(t),\textbf{n}(t),\textbf{v})\ \tau(t)\ dt\\&\approx\sum_{i=1}^{m-1}\delta_i\ \tau(s_i)\ L_i\\ &= \sum_{i=1}^{m-1}\delta_i\ \sigma(s_i)\ T(s_i)\ L_i \end{aligned}\tag{7}$$接着将$(3)$化为离散形式（黎曼和）：$$T(s_i)\approx\widehat{T}(s_i)=exp(-\sum_{j=1}^{i-1}\sigma_j\ \delta_j)\tag{8}$$为了将**离散的**概率密度分布 $\widehat{\tau}$ 来代替原本连续的 $\tau$（这样就能够得到式子 $(7)$ 的准确得离散表达形式）。这里设 $p_i=exp(-\sigma_i\delta_i)$ ，根据式子$(3)$可以理解为当前采样点的区间 $[s_i,s_{i+1}]$ 的透明度，于是可以将$(7)$改写成 $T(s_i)=\prod_{j=1}^{i-1}p_i$ 表示整个光线区间的透明度；现在式 $(7)$ 中只剩下 $\delta_i\sigma(s_i)$ 需要处理，因为区间$\delta_i$很小，且体密度$\sigma(s_i)$小于1，乘起来后较小，于是可以近似地用$p_i$来表示，即近似为 $\delta_i\sigma(s_i)\approx(1-exp(-\delta_i\sigma_i))=1-p_i$；最终得到式子$(7)$的更符合数学上的准确表达：$$\begin{aligned}I(\textbf{c},\textbf{v})\approx\widehat{I}(\textbf{c},\textbf{v})&=\sum_{i=1}^{m-1}\ [(1-p_i)\ \prod_{j=1}^{i-1}p_j]\ L_i\\&=\sum_{i=1}^{m-1}\widehat{\tau}_i\ L_i\end{aligned}\tag{9}$$其中 $\widehat{\tau}_i=(1-p_i)\prod_{j=1}^{i-1}p_j$ 表示的是**不透明度的离散形式的概率**（$\tau_i$ 为原本的连续形式的概率密度）

### 3. 采样

#### 1. NeRF的不足 以及 采样策略的提出

根据 $(5)(6)$ 式可以知道该体渲染的积分相当于radiance field $L$ 在不透明度的概率密度函数 $\tau(t)$ 下的期望值/积分；由于 PDF函数 $\tau(t)$ 常常集中在物体的**边界区域/表面**（有点像高斯函数一样两边低，中间高）， 所以如果用离散采样来近似积分，那么需要针对PDF值变化较大的地方有着较多的采样，即根据PDF的特征来采样（想象一下用矩形区域来近似积分面积，那么对于变化较平滑的地方可以直接用一个高值乘以较宽的长度，即采样密度较小，就可以很好地近似这块积分区域的面积；但是对于变化较大的地方就需要更小的宽度，即更多的采样，来近似这块积分区域的面积）

所以NeRF中就用权重 $w$ ，即这里的不透明度的PDF函数 $\tau(t)$ 来得到对应的CDF函数，并利用 inverse CDF $O^{-1}$ 来作适应性采样（即NeRF的fine采样）。但是这种方法依赖于CDF $O$ 的准确性，而$O$又依赖于预测的 $\sigma$ 的准确性；1）本来从coarse网络中**均匀采样**得到的 $\sigma$ 就不够准确，根据这些 $\sigma$ **得到的 $O$ 也就不够准确**，同时2）在fine网络中学习到的 $\sigma$ 又和coarse网络中的不同，也就是fine网络的 $O$ 与 coarse网络的 $O$ 也有所不同，这两种误差结合起来导致NeRF产生sub-optimal次优得样本集 $S$ 以及缺少/过渡扩展不可忽略的 $\tau$ 值，使得计算离散的结果不够准确

* **根本目的**：在一定采样点数量的情况下，能够**更好地根据PDF来进行采样**，从而使得离散形式与积分形式的误差更小。但是我们不知道PDF，所以就先根据误差边界来**让 $\widehat{O}$ 恢复得足够像 $O$**（即误差边界 < $\epsilon$），这样就能让根据**CDF逆变化采样**（对$\widehat{O}^{-1}$采样）得到的点更接近其PDF的分布（简单来说就是我要让采样点分布更准确，那就先还原出准确得分布，再进行采样）

于是这里就采用基于**下面就分析Opacity $O$ 的 error bound误差边界来还原出不超过一定误差 $\epsilon$ 的 $\widehat{O}$**，这样再通过对 $\widehat{O}^{-1}$ 采样得到更准确的样本集 $S$

#### 2. O(t)的误差上界的分析

为了分析积分形式的 $O(t)$ 和离散形式的 $\widehat{O}(t)$ 的误差，先将 $O(t)$ 中的积分部分提出来：$$\int_0^t\sigma(\textbf{x}(s))\ ds=\widehat{R}(t)+E(t),\quad where\ \widehat{R}(t)=\sum_{i=1}^{k-1}\delta_i\sigma_i+(t-t_k)\sigma_k\tag{10}$$其中$\widehat{R}(t)$为积分的离散近似值，$E(t)$为积分得误差。然后就可以得到离散的opacity表达式： $$\widehat{O}(t)=1-exp(-\widehat{R}(t))\tag{11}$$

1. 推导出在任意的一个积分区域 $[t_i,t_{i+1}]$ 中 $|\frac{d}{ds}\sigma(\textbf{x}(s))|$ 存在上界（这个上界只与$|d_i|$、$|d_{i+1}|$、$\alpha$、$\beta$ 参数有关）![[Pasted image 20230819163848.png]]
2. 从而推导出 $\int_0^t\sigma(\textbf{x}(s))ds$ 的误差 $E(t)$ 存在上界![[Pasted image 20230819163921.png]]
3. 最终带入式 $(4),(11)$ 可得到 $O(t)$ 的误差上界![[Pasted image 20230819164433.png]]
4. 由于 $\widehat{R}(t)$ 和 $\widehat{E}(t)$ 都是递增的（看式$(10)$的积分）所以$exp(-\widehat{R}(t))$ 递减，$(exp(\widehat{E}(t))-1)$ 递增，所以可以得到一个区间内的最大误差的上界![[Pasted image 20230819165123.png]]
5. 所以可以得到所有区间 $t\in[0,M]$ 的 $O(t)$ 的误差上界![[Pasted image 20230819165258.png]]

![[Pasted image 20230819165446.png]]

上图绿色整个区间中$O(t)$的为误差上界 $\epsilon$（即每个小区间中最大的误差上界），深红色为实际误差，浅红色为每个采样区间的误差上界

#### 3. 降低误差上界 $B_{\tau,\beta}$ 的方法

1. **充足的采样**可以降低误差边界 $B_{\tau,\epsilon}$（采样点越多就能够更加近似积分）
	* 固定 $\beta>0$，对任意 $\epsilon>0$，足够多的采样 $T$ 就能够让 $B_{\tau,\beta}<\epsilon$
2. 对于固定的采样数量 $T$，可以让 **$\beta$ 足够大**来降低误差边界 $B_{\tau,\epsilon}$（$\beta$越大，2中的$\widehat{E}(t)$越小，5中的误差上界$B_{\tau,\beta}$越小）
	* 固定 $n>0$，对任意 $\epsilon>0$，足够大的 $\beta$ 满足 $\beta\geq\frac{\alpha M^2}{4(n-1)log(1+\epsilon)}$ 就可以让 $B_{\tau,\beta}\leq\epsilon$

#### 4. 具体操作步骤

**思路分析**：

1. 从 $(2)$ 式中可以看出 $\beta$ 主要影响的是体密度$\sigma$ 和opacity不透明度 $O$ 的变化程度（函数的斜率）： $\beta$ 越小，$\sigma$ 变化越大，$O$ 变化也越大，会使得在接近物体表面的时候不透明度会瞬间增加；但是**不同类型的物体表面的变化程度不一样，也就是对应的 $\beta$ 不同，所以需要让 $\beta$ 自己去学习**（应该是初始化一个值，然后给上梯度让他去学习吧）（但是本paper中只用了一个 $\beta$，即作为式 $(2)$ 中用来得到整个场景的体密度计算函数，那我是不是可以通过一小段网络来用位置 $\textbf{x}$ 作为输入，然后输出对应位置的 $\beta$ 呢💡，这样就可以学习场景中不同位置的 $\beta$ 了）。
2. 对于当前的 $\beta$，首先需要**通过一种不断优化的采样方式来得到一个误差上界 $B_{\tau,\beta+}$不会超过给定的 $\epsilon$（一个超参） 的 $\widehat{O}$** ，同时让需要优化的 $\beta+$ 不断接近 $\beta$ （因为最终的采样的误差是根据 $\beta$ 来计算的，如果$\beta+$ 很大，与$\beta$差很远，那么$B_{\tau,\beta+}$是靠采样点数量来降低的，在最后实际采样的时候，根据$\beta$来计算的误差就会很大）。
3. 当 $\widehat{O}$ 模拟的足够准确了，那么后面根据**CDF进行逆变换采样 $\widehat{O}^{-1}$ 得到的采样点**就能更加准确

**恢复 $\widehat{O}$ 和逆变换采样的算法：

😖为什么不直接简单一点**直接采样多个点**来恢复呢？因为对于每一个采样点都需要通过一整个网络来预测SDF值 $d_\Omega(\textbf{x})$，然后再经过式 $(2)$ 才能得到体密度 $\sigma$，而需要采样大量的才能恢复出较为准确得 $\widehat{O}$；既然这样，就从误差分析出发，直接**根据误差产生的原因来减少误差**，这样效率更高

1. Input 为超参数 $\epsilon>0$ 和 当前的 $\beta$
2. 均匀采样 $T=T_0$（128） 个点 
3. 初始化一个 $\beta_+>\beta$ 来使得 $B_{\tau,\beta_+}\leq\epsilon$ （根据[[#3. 降低误差上界 $B_{ tau, beta}$ 的方法|降低误差界限方法2]]得到 $\beta$ 越大，误差界限越小）
4. **while** $B_{\tau,\beta}>\epsilon$ 且 不满足最大迭代次数（5次）（如果当前$\beta$下采样$\tau$个点就能满足误差界限够小了，那么最后的采样后的误差也不会太大，也就不需要继续采样了）
	1. 根据所有区间的误差占比继续采样 n 个点（为了降低$B_{\tau,\beta_+}$，因为后面会减小$\beta_+$而增大误差）（在恢复 $\widehat{O}$ 过程中的采样点输入MLP中都是 $torch.no\_grad()$）
	2. **if** $B_{\tau,\beta_+}<\epsilon$
		1. 二分查找 $\beta_*\in(\beta,\beta_+)$ 使得 $B_{\tau,\beta_*}=\epsilon$
		2. 更新 $\beta_+\leftarrow\beta_*$（让$\beta_+$减小，更接近 $\beta$，使得最后$\beta$ 采样的误差不会太大）
5. 用已有的 $T,\beta_+$ 来估计 $\widehat{O}$（$\beta_+$的估计一定小于误差界限）
6. 根据 $\widehat{O}^{-1}$ 来逆变换采样m（64）个点（在最后逆变换采样中是需要梯度的（需要更新网络参数））

![[Pasted image 20230819195527.png|300]]

![[Pasted image 20230819200540.png]]
![[Pasted image 20230819200531.png]]

上图上面部分可以看出随着上面的 Sampling Algorithm地进行，$\beta_+$不断减小（靠近$\beta$）而且也从原来的较多artifacts到后面的更加准确细致。下面部份蓝色表示真实的opacity，黄色表示近似的opacity $\widehat{O}$，黑色的表示SDF，点表示对$\widehat{O}^{-1}$ 的逆变换采样点；可以看到随着迭代近似的 $\widehat{O}$ 越来越准确，同时逆变换采样的采样点也更集中在opacity 变化较大的地方（需要更多采样的地方）

## 4. Training

1. 根据[[#3. 采样]]中的方法来进行采样
2. 将采样点位置信息 $\textbf{x}$ 放到第一个MLP $\textbf{f}_\phi$ 中得到SDF和中间层 $(d(\textbf{x}),\textbf{z}(\textbf{x}))\in\mathbb{R}^{1+256}$ 
3. 这里令 $\alpha=\beta^{-1}$，同时让SDF为：$d_\Omega(\textbf{x})=min\{d(\textbf{x},r-||\textbf{x}||_2)\}$（r为近景的球半径，参考[[NeRF++-2020]]），将SDF代入式$(2)$得到体密度 $\sigma$
4. 将位置信息 $\textbf{x}$、法线 $\textbf{n}$（$\textbf{n}=\nabla_{\textbf{x}}d_\Omega(\textbf{x}(t))$，梯度为值增加最快的方向，$d_\Omega$增加最快的方向就是垂直交点平面向外的方向，就是法线方向）、观察方向 $\textbf{v}$、中间层 $\textbf{z}$ 传入第二个MLP $L_\psi$  得到场景的radiance field $\textbf{c}\in\mathbb{R}^3$ 
5. 将体密度 $\sigma$、采样点颜色 $\textbf{c}$ 带入体渲染方程式 $(9)$ 中得到像素的颜色
6. 计算损失函数![[Pasted image 20230819202539.png]]其中$L_{RGB}$为与GT的差异，$L_{SDF}$是Eikonal loss为了让 $d$ 能够更好地往Signed Distance Function方向学习

## 5. Contribution & Limitations

### 1. Contribution

1. 通过用一个**几何函数SDF来建模得到体密度 $\sigma$**，从而将原本的MLP预测$\sigma$取代为MLP预测SDF，使得体密度的计算更加准确
2. 从数学误差界限的角度详细分析了**采样**的策略（用误差界限分析得到最终采样点来代替原本NeRF中的coarse-to-fine采样策略）

### 2. Limitations

**不同类型的物体表面的变化程度不一样，也就是对应的 $\beta$ 不同，所以需要让 $\beta$ 自己去学习**，但是本paper中只用了一个 $\beta$，即作为式 $(2)$ 中用来得到整个场景的体密度计算函数，那我是不是可以通过一小段网络来用位置 $\textbf{x}$ 作为输入，然后输出对应位置的 $\beta$ 呢💡，这样就可以学习场景中不同位置的 $\beta$ 了