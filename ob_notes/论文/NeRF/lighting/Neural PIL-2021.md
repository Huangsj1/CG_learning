# Neural-PIL：Neural Pre-Integrated Lighting

## 1. motivation

1. 对于NRF和NeRV中需要**已知光源信息**
2. 而NeRD中用SG来模拟光源，这种SG / SH方法对**高频光源表征能力较差**（除非增加阶数来使得函数近似更准确，但这需要大量存储和计算资源）

## 2. Method

Neural-PIL 主要考虑的是**光源**问题，基于[[Image Based Lighting]]思想，将Render过程中需要积分计算的步骤简化：用embedding来表示每张图片的环境贴图，然后通过MLP网络来查询对应光照（Imgae-Based Lighting中通过已知的环境贴图预计算每个方向、粗糙程度的光照值并存储起来，需要用的时候直接查询即可；这里的embedding类似于已知的**环境贴图**，MLP类似于预计算每个方向、粗糙程度对应的光照值的**积分方法/函数**，有了积分函数的近似，就可以直接传入embedding和方向、粗糙度来得到需要的光照值，而不需要存储预计算值再查询）

用到的NeRF中的体渲染方程：$$C(\textbf{r}) = \int_{t_n}^{t_f}T(t)\cdot\sigma(\textbf{r}(t))\cdot\textbf{c}(\textbf{r}(t),\textbf{d})dt\tag{1}$$其中$\textbf{c}(\textbf{r}(t),\textbf{d})=L_o(\textbf{x},\pmb\omega_o)$分解得到有关BRDF的项：$$L_o(\textbf{x},\pmb\omega_o)=\int_\Omega f_r(\textbf{x},\pmb\omega_i,\pmb\omega_o)\ L_i(\textbf{x},\pmb\omega_i)\ (\pmb\omega_i\cdot\textbf{n})\ d\pmb\omega_i\tag{2}$$将$L_o(\textbf{x},\pmb\omega_o)$拆成diffuse和specular项，并用UE4中的近似方法来表示：$$\begin{aligned}L_o(\textbf{x},\pmb\omega_o)&=\frac{\textbf{b}_d}{\pi}\int_\Omega L_i(\textbf{x},\pmb\omega_i)\ (\pmb\omega_i\cdot\textbf{n})\ d\pmb\omega_i\ +\ \int_\Omega f_s(\textbf{x},\pmb\omega_i,\pmb\omega_o;\textbf{b}_s,b_r)\ L_i(\textbf{x},\pmb\omega_i)\ (\pmb\omega_i\cdot\textbf{n})\ d\pmb\omega_i \\ &\approx\frac{\textbf{b}_d}{\pi}\widetilde{L}_i(\textbf{n},1)\ +\ \textbf{b}_s\ (F_0(\pmb\omega_o,\textbf{n})B_0(\pmb\omega_o\cdot\textbf{n},b_r)+B_1(\pmb\omega_o\cdot\textbf{n},b_r))\ \widetilde{L}_i(\pmb\omega_r,b_r) \end{aligned}\tag{3}$$式子中 $\widetilde{L}_i(\pmb\omega_r,b_r)=\int_\Omega D(b_r,\pmb\omega_i,\pmb\omega_r)\ L_i(\textbf{x},\pmb\omega_i)\ d\pmb\omega_i$ 同时近似表示了[[Image Based Lighting]]中的**漫反射部分**和**镜面反射部分的左边部分**，其中漫反射部分传入的是法线和粗糙度为1，镜面反射部分传入的是反射向量和粗糙度$b_r$，用同一个式子来近似本来需要两个预训练的计算，也就相当于用同一个网络Neural-PIL来学习这两个预计算的积分函数。而$B_0(\pmb\omega_o\cdot\textbf{n},b_r)$表示的是预计算的A，$B_1(\pmb\omega_o\cdot\textbf{n},b_r)$表示的是预计算的B。所以最终$L_o$可以通过采样点的信息来直接通过网络和预计算的结果来得到

## 3. Procedure

![[Pasted image 20230817103453.png]]

整体分为两个大网络：Coarse Network 和 Decomposition Network，大体步骤类似于[[NeRD-2020]]

### 1. Coarse Network

>主要负责提供采样点权重信息给后者采样

整个网络流程和[[Wild NeRF-2021]]的static网络类似，先由位置Position $\textbf{x}$ 输入得到体密度 $\sigma$，然后输入View Direction $\pmb\omega_o$ 和每张图片各自的Illumination Embedding $\textbf{z}^l$ （128维，类似于W-NeRF中的每张图片的latent code，只不过W-NeRF中学到只是图片的风格，没有具体物理含义，没法用在渲染公式中；这里代表的是环境贴图的信息，可以经过网络查询得到光照信息用在渲染公式中），得到每个采样点的Color，并通过体渲染方程$(1)$得到每个采样点的颜色，然后和GT进行L2损失更新参数

这里并没有更新每张图片的环境光照Illumination Embedding $\textbf{z}^l$，而是在后面的Decomposition Network中才更新

### 2. Decomposition Network

>主要负责学习提取体密度 $\sigma$、BRDF信息$\textbf{d}$、法线信息$\textbf{n}$、每张图片光照信息$\textbf{z}^l$

#### 1. 预训练部分

![[Pasted image 20230817104958.png]]

##### 1. Smooth manifold auto-encoder

原因：因为一起估计光照、材质信息是非常underconstrained欠约束的事情，为了得到可信的结果，需要对Illumination和BRDFs进行**regularize正则化**。于是这里通过学习一个**low-dimensional smooth manifolds低维光滑流形的auto-encoder**来将原本高维的信息转成**低维**来得到更本质的特征（可参考[manifold介绍](https://blog.csdn.net/a493823882/article/details/115433888)），使得Illumination和BRDFs都能得到更好的学习和优化

具体步骤：这里预训练两个SMAE：

1. **BRDF-SMAE**：因为BRDF所需要的参数 $\textbf{b}$ 为7维的（3维diffuse $\textbf{b}_d$，3维specular $\textbf{b}_s$，1维roughness $b_r$），所以input $\textbf{p}$ 为7维的，中间编码 $\textbf{z}^b$ 为4维的。训练用的是已有的一些dataset，将训练后的**decoder后半部分用于Decomposition Network中的 ”BRDF SMAE“部分**，将前面网络输出的4维 $\textbf{z}^b$ decode 成需要的7维的 $\textbf{b}$
2. **Light-SMAE**：数据集中的environment maps都是128 * 256维的，将其压缩成128维的smooth latent space $\textbf{z}^l$ 来训练。训练好的auto-encoder不是用在Decomposition Network中，而是**用来训练Neural-PIL**部分

##### 2. Neural-PIL

用训练后的Light-SMAE中的encoder来训练Neural-PIL：对于dataset中的每个environment map，经过encoder降到128维作为Illumination embedding $\textbf{z}^l$，然后输入到Neural-PIL中。对于每个方向$\pmb\omega_r$和粗糙程度$b_r$，经过Neural-PIL后得到对应的预测光照结果，同时用[[Image Based Lighting]]中预计算的方法计算真实的光照结果（或许这个预计算结果也是预计算好的，这样可以直接查询），二者进行损失计算来优化Neural-PIL，最后**这个Neural-PIL就用在Decomposition Netwok中**

##### 3. $B_0$和$B_1$

$B_0,B_1$对应[[Image Based Lighting]]中的A和B，二者都可以预计算的到只与$cos\theta,roughness$有关的LUT贴图，使用的时候直接查询即可

#### 2. 实际渲染部分

经过预计算后，$\widetilde{L}_i(\pmb\omega_r,b_r)$ 可以在采样的时候直接查询网络Neural-PIL得到，$B_0,B_1$可以直接查询预计算的LUT贴图得到，于是 $L_o(\textbf{x},\pmb\omega_o)$ 也可以直接得到，极大减少了积分的计算量

1. 传入采样点位置 $\textbf{x}$ 经过 "Main Network" 得到 体密度$\sigma$ 和 4维的压缩了的 $\textbf{z}^b$
2. "BRDF SMAE" 为预训练的auto-encoder中的decoder，将4维的$\textbf{z}^b$ decode 成7维的BRDF $\textbf{b}$
3.  到现在为止得到的全都是 *采样点* 的各种信息，将**每个采样点的 $\alpha$ 权重来将所有采样点的属性累加起来**，最终得到当前 *光线* 的期望属性值：BRDF属性 $\textbf{b}$ 、法线 $\textbf{n}$（将得到的所有采样点的体密度 $\sigma$ 取负偏导得到采样点法线再求期望）；然后根据观察方向 $\pmb\omega_o$ 和 法线 $\textbf{n}$ 得到当前期望点的反射方向 $\pmb\omega_r$；将**期望点**的反射方向$\pmb\omega_r$、BRDF中的粗糙度$b_r$、随机初始化的环境光照Illumination Embedding $\textbf{z}^l$ 传入到 ”Neural-PIL“ 得到 $\widetilde{L}_i(\pmb\omega_r,b_r)$
4. 最后将 入射方向$\pmb\omega_o$、$\widetilde{L}_i(\pmb\omega_r,b_r)$、法线$\textbf{n}$、BRDF $\textbf{b}$、LUT中的$B_0,B_1$都传入渲染方程 "Renderer" 式$(3)$中得到最终的$L_o$，也就是Color $\textbf{c}$
5. 将预测的图像与GT进行损失计算，更新网络"Main Network" 和 Illumination Embedding $\textbf{z}^l$（得到更准确的环境贴图）。因为 "BRDF SMAE" 和 "Neural-PIL" 都是预训练好的，所以不用更新

## 4. contribution & limitations

### 1. contribution

1. 基于 Image-based Lighting，用一个**网络Neural-PIL来学习预计算的函数**，并且用类似latent code的Illumination Embedding $\textbf{z}^l$来学习每张image的环境光照，能更好反映出高光部分
2. 用SMAE来将高维信息压缩成**低维信息**，学到本质特征

### 2. limitations

1. 物体间的**间接光照**没有考虑，因为用到了环境光照贴图的想法，都是直接光照
2. relighting的话需要重新完整地光线追踪
3. 基于Imgae-based Lighting 的部分不太符合物理的建模和渲染着色方法，简化了很多过程导致**材质的建模**变得简单，难以恢复得更准确