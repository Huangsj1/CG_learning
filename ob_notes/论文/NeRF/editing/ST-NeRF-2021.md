# Editable Free-Viewpoint Video using a Layered Neural Representation

> 通过为每个人和背景构建一个**bounging-box以及对应的ST-NeRF**，分层学习每个bounding-box相关的时间、空间信息，渲染的时候对所有的bounding-box一起渲染，编辑的时候只用编辑对应的bounding-box

![[Pasted image 20230822143907.png]]

## 1. Scene Parsing 场景解析

![[Pasted image 20230822144039.png|400]]

1. **Coarse depth**：使用 Multi-view stereo [P-MVS](https://zhuanlan.zhihu.com/p/341591068)方法得到场景的**深度**信息图
2. **Low resolution mask**：人为给定初始帧的所有bounding boxes，接着使用 [SiamMask tracker](https://zhuanlan.zhihu.com/p/58154634)得到**所有的bounding boxes后续的bounding boxes和masks**（$G_{t_1,t_2}^c=\{g_t^c\in\mathbb{R}^2\}_{t=t_1}^{t_2}$，其中 $g_t^c$ 为c方向t时刻的bounding box的中心）；但是该tracker方法对于物体occlusion重叠阻塞情况预测得不好，所以这里又添加了 [trajectory prediction network (TPN)](https://zhuanlan.zhihu.com/p/139905513)轨迹预测网络$\Theta_{TP}(\cdot,\cdot)$ 来**正则化预测的结果**：$g_t^{b'}=q^bg_t^b+\frac{1-q^b}{w}\sum_{c,q^c\geq\tau}q^c\Theta_{TP}(G^c_{t_0,t},G^b_{t_0,t_1})$；最终利用所有的2D bounding boxes和masks**得到所有的3D bounding boxes 和 人体几何**
3. **Refined mask**：因为2中预测的masks在人物重叠部分预测的不够准确，所以这里采用一种方法来改进：假设前一帧的masks准确（因为第一帧是人为给定的bounding boxes，所以也可以给定masks），那么就根据1中的深度图先记录每个人/bounding box的平均深度 ${D_t^c}$（前一帧的），在当前帧对于每个人的mask对应的位置，如果**深度图中的深度和前一帧的平均深度差别很大就直接抛弃**

## 2. Spatio-Temporal Neural Radiance Field 时空神经辐射场

![[Pasted image 20230822155136.png]]

对于每一个bounding box 在**不同视角不同时间的图片都用单独用一个 ST-NeRF 的网络来学习**，具体参考的方法是[D-NeRF](https://arxiv.org/abs/2011.13961)中处理动态场景的方法。

ST-NeRF包含两个网络：**space-time deform module时空编写模块 $\phi^d$ 和 neural radiance module神经辐射模块 $\phi^r$**；前者表示为：$$\Delta p=\phi^d(p,t,\theta^d)\tag{1}$$传入采样点位置p和时间t到网络中（网络参数为$\theta^d$）得到当前时刻t该位置p相对初始位置偏移的位置信息 $\Delta p$；将偏移后的位置信息 $p+\Delta p$ 传入到网络 $\phi^r$ 中得到采样点的颜色和体密度：$$(c,\ \sigma)=\phi^r(p+\Delta p,\ d,\ t,\ \theta^r)\tag{2}$$因为这两个网络是前后相连的，所以可以统一写为：$$(c,\sigma)=\phi(p,d,t,\Theta)\tag{3}$$

## 3. Network Training 网络训练

![[Pasted image 20230822162921.png]]

1. **coarse网络的采样**：对于当前时刻 t 下的当前光线 $r(s)$，计算其**与所有包围盒 $B_t^i$ 的交点** $S^i=\{s_n^i,s_f^i|s_n^i<s_f^i\}$，然后在**有交点的包围盒内均匀采样**，得到coarse网络的第i个物体的所有N个采样点 $P_c^i=\{r(s_j^i)|j\in[1,2,...,N])\}$
2. **采样点的排序**：得到coarse网络中所有包围盒的所有采样点后，需要根据**深度从近到远重新排序所有采样点** $P=\bigcup_{i\in I}P^i$
3. **采样点权重计算与体渲染**：计算coarse网络的所有采样点的权重$w$，然后根据体渲染方程得到当前光线的颜色：$$\begin{aligned}\widehat{C}(\textbf{r})&=\sum_{j=1}^{|P|}T(p_j)[(1-exp(\sigma(p_j)\delta(p_j)))\ c(p_j)] \\ T(p_j)&=exp(-\sum_{k=1}^{j-1}\sigma(p_k)\delta(p_k))\end{aligned}\tag{4}$$
4. **fine网络的采样与体渲染**：根据3中所有采样点的权重进行**逆变换采样**（同NeRF的方法），之后同式$(4)$一样计算光线的颜色
5. **损失函数计算**：损失函数除了预测的图像与GT的差异$L_{rgb}$，同时假设位于对于有遮挡的物体，**前面的物体是不透明**的，也就是鼓励前面的物体的不透明度为1，后面被遮挡的物体的不透明度为0，从而加快网络的训练（相当于加强限制从而使得网络更容易学习）：$$\begin{aligned}L&=(1-\lambda)L_{rgb}+\lambda L_{layer}\\&=(1-\lambda)\sum_{r\in R}(||C(r)-\widehat{C}_c(r)||^2_2+||C(r)-\widehat{C}_f(r)||^2_2)\\&+\lambda\frac{1}{2}\sum_{i=1}^{n_i}||\Omega(r,L,i)-\alpha(r,i)||^2_2\end{aligned}$$其中 $\Omega(r,L,i)$ 是一个指示函数：如果当前光线r在label map中相交的最前面的那个包围盒为第i个包围盒，那么就是1，否则就是0；$\alpha(r,i)=\sum_{j=1}^{|P^i|}exp(-\sum_{k=1}^{j-1}\sigma(p_k)\delta(p_k))(1-exp(-\sigma(p_j)\delta(p_j)))$即不透明度opacity（也可以当成是权重）

## 4. Scene Editing 场景编辑

![[Pasted image 20230822122720.png]]

1. **空间上的放射变换**：通过**对bounding-box $B^i$ 进行空间上仿射变换 $A$** 后得到新的bounding-box $\widehat{B}^i=A\circ B^i$，但是网络没有变，学习到的依然是原本bounding-box $B^i$ 所对应的场景，所以在对新的 $\widehat{B}^i$ 采样的时候，需要将采样点的位置p和方向d都进行逆仿射变换后（得到原本$B^i$位置）再传入ST-NeRF网络中：$$\phi(A^{-1}\circ \textbf{p},\ A^{-1}\circ \textbf{d},\ t,\ \Theta^i)=(\textbf{c},\sigma)$$
2. **时间上的仿射变换**：与空间类似，不过这里是**对时间 $t$ 进行仿射变换 $T$** 从而得到不同的时间线：$$\phi(\textbf{p},\ \textbf{d},\ T\circ t,\ \Theta^i)=(\textbf{c},\sigma)$$
3. **物体插入和删除**：因为这里的渲染是通过对所有的bounding-box来求交点采样的，所以需要**添加/删除对应的bounding-box**，但是添加的话还需要添加该bounding-box对应的其他东西（如tracker、ST-NeRF等）
4. **透明度调整**：因为渲染是对所有bounding-box来采样得到采样点并传入对应的ST-NeRF来预测颜色$\textbf{c}$和体密度$\sigma$的，所以可以直接对需要透明化的bounding-box对应的采样点的体密度$\sigma$进行调整：$$\sigma'=s\cdot\sigma$$
## 5. Contributions & Limitations

### Contributions

1. **Layered Neural Representation**：将物体（人）分成**单独的bounding-box**来处理，为每个bounding-box训练单独的ST-NeRF，里面包含了对**空间和时间**的学习，于是可以对每个物体单独地进行时间/空间上的编辑

### Limitations

1. Insertion 或者 Removal 后对于**光影**肯定不行
2. 这里只是对人体进行bounding-box的包围（除了人体之外就全部当成背景的bounding-box），也就是**只适用于人体的变换**（如果能够学习到对所有物体的分割就更好）
3. 对于**过多的障碍**会使得bounding-box的轨迹跟踪更加困难（这一块应该是物体轨迹追踪这些方向需要解决的问题）
4. 因为这里集中处理的是bound-box，对于**bounding-box之外的属性**（如光照、阴影等）无法处理