# NeRFactor：Neural Factorization of Shape and Reflectance

![[Pasted image 20230817191142.png|400]]

## 1. motivation

与Neural-PIL同一时期的工作，所以动机差不多

1. 对于NRF和NeRV中需要**已知光源信息**
2. 而NeRD中用SG来模拟光源，这种SG / SH方法对**高频光源表征能力较差**

## 2. Method

与[[Neural PIL-2021]]相同都是用Environment Mapping环境贴图来模拟光照（但是Neural PIL侧重的是学习预计算光照的函数，这里侧重的是用一个经纬度贴图来学习环境贴图）；不同的是这里更注重**BRDF学习**（Neural PIL更注重光照的学习），引入BRDF库预训练来得到先验条件更好地建模BRDF

## 3. Procdure

![[Pasted image 20230817191207.png]]

这里分了**3个不同的阶段**来得到三种不同类型的网络：预训练好的固定的网络（蓝色）、与训练好的需要微调的网络（绿色）、从头开始训练的网络（黄色）

渲染的时候采用的是同Neural PIL和NeRD中render类似的方法：都是对期望采样点来进行渲染（不同NeRF是对所有采样点渲染再积分），其中期望采样点求法为：$$\textbf{x}_{surf}=\textbf{o}+[\int_0^\infty T(t)\ \sigma(\textbf{r}(t))\ t\ dt\ ]\ \textbf{d}\tag{1}$$渲染方程为：$$L_o(\textbf{x}_{surf},\pmb\omega_o)=\int_\Omega \textbf{R}(\textbf{x}_{surf},\pmb\omega_i,\pmb\omega_o)\ L_i(\textbf{x}_{surf},\pmb\omega_i)\ (\pmb\omega_i\cdot\textbf{n})\ d\pmb\omega_i\tag{2}$$其中完整的BRDF可以进一步拆成漫反射和镜面反射部分：$$\textbf{R}(\textbf{x}_{surf},\pmb\omega_i,\pmb\omega_o)=\frac{\textbf{a}(\textbf{x}_{surf})}{\pi}+\textbf{f}_r(\textbf{x}_{surf},\pmb\omega_i,\pmb\omega_o)\tag{3}$$

### 1. Pre-trained; Frozen Network

#### 1. NeRF（1st half）

同NeRF中训练方法一样得到一个同NeRF一样的网络，然后取前半部分（生成 $\sigma$ 的部分）作为第一个网络来得到**几何信息 $\sigma$

#### 2. BRDF MLP

BRDF中的 $\textbf{a}(\textbf{x}_{surf})$ 即albedo可以通过一个网络“Albedo MLP”学习，这里的完全预训练的 “BRDF MLP” 是为了学习**镜面反射的BRDF $\textbf{f}_r(\textbf{x}_{surf},\pmb\omega_i,\pmb\omega_o)$**

**理由**：直接通过一个网络学习BRDF的 $f_r$不够好（可能学不到或者学得很慢），需要加入**先验learning prior**或者**限制条件**才能准确快速学到BRDF $f_r$

**思路**：如果能够学习一个真实世界的所有BRDFs的 **”latent space“** 和一个能将该latent space中的 latent code 解码为原本的4D的BRDF的 **”decoder“**，那么就可以将这个 “decoder” 放到网络中作为先验/约束使得前面的网络能够学习得到当前场景的latent code，并且能够经过 “decoder” 得到BRDF $\textbf{f}_r$（把世界上所有材质的BRDF投射到一个隐藏空间中，每种材质的BRDF都可以用一个独特的latent code来表示，然后将其输入到对应的”decoder“中来得到不同方向的BRDF $f_r$）

**实际方法**：这里用[[GLO-lantent code]]方法在MERL dataset中预训练”decoder“作为frozen固定的”BRDF MLP“，这样在实际过程中只用学习一个能够生成正确的latent code的网络，即在实际训练过程中从头开始训练一个 “BRDF Identity MLP”网络来学习生成当前场景的latent code $\textbf{z}_{BRDF}$，将当前场景的 $\textbf{z}_{BRDF}$ 传入到 “BRDF MLP” 中（并传入与入射、反射、法线相关的参数）就可以解码出其真实的BRDF $f_r$。因为“BRDF MLP” 这个解码器是固定的，为了渲染时得到更准确的color，“BRDF Identity MLP”就需要往能够生成更准确的latent code $\textbf{z}_{BRDF}$ 方向去学习；而且**低维的latent code $\textbf{z}_{BRDF}$ 能够学习高维空间中的流形的本质特征**，可以加快学习过程

镜面反射网络 $\textbf{f}_r$ 除了需要对应的latent code $\textbf{z}_{BRDF}$，还需要输入 $(\textbf{n},\pmb\omega_i,\pmb\omega_o)$，这里运用一个函数 $\textbf{g}:((\textbf{n},\pmb\omega_i,\pmb\omega_o))\rightarrow(\phi_d,\theta_h,\theta_d)$（运用Rusinkiewicz 坐标系减少输入参数），同时将修改了输入参数的反射网络以下面形式表示：$$\textbf{f}_r':(\textbf{z}_{BRDF},(\phi_d,\theta_h,\theta_d))\rightarrow\textbf{r}\tag{4}$$上式就是最终的 “BRDF MLP” 网络表示形式，其中结果 $\textbf{r}$ 就是点对应方向的式$(3)$中的$\textbf{f}_r$

### 2. Pre-trained; jointly finetuned Network

根据NeRF网络可以得到所有采样点的体密度 $\sigma$，之后可以通过式$(1)$得到硬表面的期望点$\textbf{x}_{surf}$

#### 1. Normal MLP

根据采样点的权重可以得到期望点的法线 $\textbf{n}_a(\textbf{x}_{surf})$（这里是根据采样点权重得到法线还是将期望点输入NeRF中的MLP得到体密度再得到法线？）；但是由于直接从NeRF中得到法线（体密度的负梯度）会**有较大的noisy噪点**，于是通过一个Normal MLP来得到 $\textbf{x}_{surf}$ 的更加准确光滑的法线：$$\textbf{f}_n:\textbf{x}_{surf}\rightarrow\textbf{n}\tag{5}$$因为需要得到的法线不仅要1）和原本从NeRF中得到的**法线相近**，而且要2）在3D空间中**较平滑**，不会太多突变/噪点，所以构建了下面的损失函数：$$\begin{aligned}l_n=\sum_{\textbf{x}_{surf}}(\frac{\lambda_1}{3}||\textbf{f}_n(\textbf{x}_{surf})-\textbf{n}_a(\textbf{x}_{surf})||^2_2\\+\frac{\lambda_2}{3}||\textbf{f}_n(\textbf{x}_{surf})-\textbf{f}_n(\textbf{x}_{surf}+\epsilon)||_1)\end{aligned}\tag{6}$$

预训练的时候将该网络放到训练好的NeRF后面，让NeRF中得到的法线结果和经过这个网络得到的结果相同即可

#### 2. Light Visibility MLP

这里需要得到的是期望点 $\textbf{x}_{surf}$ 到光源的可视度visibility（也即透射率Transmittance），预训练的时候给定512个固定的已知光源点（环境贴图），于是可以通过光线方向 $\pmb\omega_i$ 来进行ray marching得到采样点的体密度 $\sigma$，积分得到NeRF对应的visibility $v_a(\textbf{x}_{surf},\pmb\omega_i)$；但是直接从NeRF中得到的visibility同法线一样会有较大的**noisy噪点**，于是通过一个Light Visibility MLP来得到更准确的visibility：$$f_v:(\textbf{x}_{surf},\pmb\omega_i)\rightarrow v\tag{7}$$因为需要得到的visibility不仅要1）和原本从NeRF中得到的**visibility相近**，而且要2）在3D空间中**较平滑**，不会太多突变/噪点，所以构建了下面的损失函数：$$\begin{aligned}l_v=\sum_{\textbf{x}_{surf}}\sum_{\pmb\omega_i}(\lambda_3(\textbf{f}_v(\textbf{x}_{surf},\pmb\omega_i)-v_a(\textbf{x}_{surf},\pmb\omega_i))^2\\+\lambda_4|\textbf{f}_v(\textbf{x}_{surf},\pmb\omega_i)-\textbf{f}_v(\textbf{x}_{surf}+\epsilon,\pmb\omega_i)|_1)\end{aligned}\tag{8}$$

预训练的时候将该网络放到训练好的NeRF后面，让NeRF中得到的visibility结果和经过这个网络得到的结果相同即可

### 3. Trained from scratch Network

预训练 ”Normal MLP” 和 “Light Visibility MLP” 后再进行jointly finetuned微调，防止 "Albedo MLP" 和 ”BRDF Identity MLP“ 将阴影误认为”painted on“本身就在上面的反射属性而导致Albedo 和 BRDF Identity学不好

#### 1. Albedo MLP

这里对也是通过一个网络来学习反照率Albedo：$$\textbf{f}_a:\textbf{x}_{surf}\rightarrow\textbf{a}\tag{9}$$不过这里不像上面的网络一样预训练，这个网络约束较少（其他网络约束较好，防止该网络学走其他东西），只是为了让学到的albedo更平滑（同Normal MLP 和 Light Visibility MLP一样防止出现噪点）：$$l_a=\lambda_5\sum_{\textbf{x}_{surf}}\frac{1}{3}||\textbf{f}_a(\textbf{x}_{surf})-\textbf{f}_a(\textbf{x}_{surf}+\epsilon)||_1\tag{10}$$

#### 2. BRDF Identity MLP

从[[#2. BRDF MLP]]的介绍中可以知道这个网络的目的是学习当前点的有关镜面反射的BRDF的latent code隐藏信息：$$\textbf{f}_z:\textbf{x}_{surf}\rightarrow\textbf{z}_{BRDF}\tag{11}$$这里除了有后面的BRDF MLP约束之外，还需要让其学得更平滑（同上面网络一样）：$$l_z=\lambda_6\sum_{\textbf{x}_{surf}}\frac{||\textbf{f}_z(\textbf{x}_{surf})-\textbf{f}_z(\textbf{x}_{surf}+\epsilon)||_1}{dim(\textbf{z}_{BRDF})}\tag{12}$$

### 4. Lighting 光照

光照使用的是16 * 32分辨率的环境贴图（经纬贴图），需要让贴图学得更加光滑：$$l_i=\lambda_7(||\left[\begin{matrix}-1&1\end{matrix}\right]*\textbf{L}||^2_2\ +\ ||\left[\begin{matrix}-1\\1\end{matrix}\right]*\textbf{L}||^2_2)\tag{13}$$

### 5. 最终过程

根据式$(3)(4)(9)(11)$可以化简得到完整的BRDF为：$$\textbf{R}(\textbf{x}_{surf},\pmb\omega_i,\pmb\omega_o)=\frac{\textbf{f}_a(\textbf{x}_{surf})}{\pi}+\textbf{f}_r'(\textbf{f}_z(\textbf{x}_{surf}),\textbf{g}(\textbf{f}_n(\textbf{x}_{surf})\pmb\omega_i,\pmb\omega_o))\tag{14}$$

并得到最终的渲染方程：$$\begin{aligned}L_o(\textbf{x}_{surf},\pmb\omega_o)&=\int_\Omega \textbf{R}(\textbf{x}_{surf},\pmb\omega_i,\pmb\omega_o)\ L_i(\textbf{x}_{surf},\pmb\omega_i)\ (\pmb\omega_i\cdot\textbf{n})\ d\pmb\omega_i \\ &=\sum_{\pmb\omega_i}\textbf{R}(\textbf{x}_{surf},\pmb\omega_i,\pmb\omega_o)\ L_i(\textbf{x}_{surf},\pmb\omega_i)\ (\pmb\omega_i\cdot\textbf{n})\ \Delta\pmb\omega_i \\ &= \sum_{\pmb\omega_i}(\frac{\textbf{f}_a(\textbf{x}_{surf})}{\pi}+\textbf{f}_r'(\textbf{f}_z(\textbf{x}_{surf}),\textbf{g}(\textbf{f}_n(\textbf{x}_{surf})\pmb\omega_i,\pmb\omega_o)))L_i(\textbf{x}_{surf},\pmb\omega_i)\ (\pmb\omega_i\cdot\textbf{f}_n(\textbf{x}_{surf}))\ \Delta\pmb\omega_i \end{aligned}\tag{15}$$其中$\Delta\pmb\omega_i$ 表示$\pmb\omega_i$ 方向的立体角（一个环境贴图中的一个像素对应立体角大小？）

渲染结果与GT的损失函数为$l_{recon}$，所有的损失函数为 $$l_{recon}+l_n+l_v+l_a+l_z+l_i$$

## 4. Contribution & Limitations

### 1. Contribution

1. 用一个**简单的环境贴图来模拟光照**，虽然没有Neural PIL那么准确，但是更加简洁方便
2. 用Visibility MLP来学习期望**点到光源的可视度/透射率**
3. 各种**预训练模型**，使得网络能够加速且正确收敛/学习，同时最重要的是**BRDF MLP**中使用现实dataset来预训练学习decoder来解码场景的latent code（**从真实世界采集信息作为先验**）
4. 对直接从NeRF得到的法线、visibility，或者间接通过网络得到的albedo、BRDF、light等都通过**smooth操作使得更加平滑**，减少noisy

### 2. Limitations

1. **没有考虑间接光照**，渲染方程中只用了一次直接光照（而NeRV中用了直接光照+一次间接光照），会使得很多该亮的地方没有亮
2. 使用的需要更新的环境映射environment mapping（具体是一个16 * 32像素的 latitude-longitude map），由于**光源分辨率问题**难以恢复硬阴影和高频的BRDFs，若增大分辨率则需要更大的计算资源
3. **依赖NeRF中生成几何的模型**来作为固定的模型，如果NeRF中生成的不好，那么会导致整个网络都不好

