# 体渲染 Volume Rendering

## 1. Procedure 过程

Ray marching 一小段一小段地走是因为 $\sigma_t$ 不同，但是假设每一小段的 $\sigma_t$ 一样，这样就可以得到当前段的 $Transmission = exp(-density\cdot \sigma_t\cdot step\_size)$；而 Forward 是因为在前进的过程中能够**累计**当前总的 $Transmission\ *=...$ 

对于每个采样点，需要得到光线到达当前点的color值，这也需要对**光线方向**进行 Ray marching（不同段有不同的密度），但是这里可以直接求总的 density来直接求总的 T（不需要用到其中某一部分来计算）$T_{light} = exp(-density_{sum}\cdot\sigma_t\cdot step\_size)$

当前采样点得到的最后的color可以分为两部分：

1. 从光源到当前采样点：$$L_i = light\_color\cdot T_{light}\cdot phase\_funct\cdot\sigma_s\cdot density\_now\cdot stride$$
	* $light\_color$ 为光源color
	* $T_{light}$ 为从光源到采样点的Transmission
	* $phase\_func$为与入射和出射夹角有关的phase function
	* $\sigma_s$为当前采样点的scatter coefficient
	* $density\_now$ 为当前采样点的密度，用于调节 $\sigma_s$（可以认为涉及到 $\sigma$ 的都需要用 $density$ 来调节）
	* $stride$ 为相机的 ray-marching 的采样的步长，用于求当前段总的color
2. 从当前采样点到相机：$$C = L_i\cdot Transmission$$
	* $L_i$ 为到当前采样点的这一段的color
	* $Transmission$ 为从 $t_0$ 到当前采样点的总的 T 透射率

![[Pasted image 20230803093042.png|300]] ![[Pasted image 20230803093112.png|300]]

```cpp
// [comment]
// This function is now called by the integrate function to evaluate the density of the 
// heterogeneous volume sphere at sample position p. It returns the value of the Perlin noise
// function at that 3D position remapped to the range [0,1]
// [/comment]
// 评估每一段的密度
float eval_density(const vec3& p)
{ 
    float freq = 1;
    return (1 + noise(p.x * freq, p.y * freq, p.z * freq)) * 0.5;
}

vec3 integrate(
    const vec3& ray_orig, 
    const vec3& ray_dir, 
    const std::vector<std::unique_ptr<Sphere>>& spheres)
{
    ...

    const float step_size = 0.1;
    float sigma_a = 0.5; // absorption coefficient
    float sigma_s = 0.5; // scattering coefficient
    float sigma_t = sigma_a + sigma_s; // extinction coefficient
    float g = 0; // henyey-greenstein asymetry factor
    uint8_t d = 2; // russian roulette "probability"

	// 定义forward ray marching 的一些参数
    int ns = std::ceil((isect.t1 - isect.t0) / step_size);
    float stride = (isect.t1 - isect.t0) / ns;

    vec3 light_dir{ -0.315798, 0.719361, 0.618702 };
    vec3 light_color{ 20, 20, 20 };

	// ray marching 方向的当前总的T
    float transparency = 1; 
    // 最终的color
    vec3 result{ 0 }; 

	// 一、沿着ray-marching方向采样
    // The main ray-marching loop (forward, march from t0 to t1)
    for (int n = 0; n < ns; ++n) {
        // Jittering the sample position
        float t = isect.t0 + stride * (n + distribution(generator));
        vec3 sample_pos = ray_orig + t * ray_dir;

        // [comment]
        // Evaluate the density at the sample location (space varying density)
        // 得到当前采样点的密度和透明度T_now
        float density = eval_density(sample_pos);
        float sample_attenuation = exp(-step_size * density * sigma_t);
        // (1)这里每一段都*=是因为当前累积的transparency需要用到
        transparency *= sample_attenuation;

        // 如果有 In-scattering.
        IsectData isect_light_ray;
        if (density > 0 && hit_sphere->intersect(sample_pos, light_dir, isect_light_ray) && isect_light_ray.inside) {
	        // 二、沿着光线方向采样
            size_t num_steps_light = std::ceil(isect_light_ray.t1 / step_size);
            float stide_light = isect_light_ray.t1 / num_steps_light;
            float tau = 0;
            
            // Ray-march along the light ray. Store the density values in the tau variable.
            // (2)这里累加所有点的密度，然后再直接求沿着光线方向总的T是因为只需要用到这一过程的总的值
            for (size_t nl = 0; nl < num_steps_light; ++nl) {
                float t_light = stide_light * (nl + 0.5);
                vec3 light_sample_pos = sample_pos + light_dir * t_light;
                tau += eval_density(light_sample_pos);
            }
            float light_ray_att = exp(-tau * stide_light * sigma_t);
            // 最后计算当前采样点的color
            result += light_color *      // light color
                      light_ray_att *    // light ray transmission value
                      phaseHG(-ray_orig, light_dir, g) * // phase function
                      sigma_s *          // scattering coefficient
                      density *          // volume density at the sample location
                      stride *           // dx in our Riemann sum
                      transparency;      // ray current transmission value 
        }

        // Russian roulette
        if (transparency < 1e-3) {
            if (distribution(generator) > 1.f / d)
                break;
            else
                transparency *= d;
        }
    }

    // combine background color and volumetric sphere color
    return background_color * transparency + result;
}
```

### noise 方程

用的是 [Perlin Noise](https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/perlin-noise-part-2/perlin-noise.html) 来生成不同的[Value Noise](https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/procedural-patterns-noise-part-1/introduction.html)来生成**不同的density**，使得物体的形状发生变化

```cpp
int p[512]; // permutation table (see source code)
 
double fade(double t) { return t * t * t * (t * (t * 6 - 15) + 10); }

double lerp(double t, double a, double b) { return a + t * (b - a); }

double grad(int hash, double x, double y, double z)
{
    int h = hash & 15;
    double u = h<8 ? x : y,
           v = h<4 ? y : h==12||h==14 ? x : z;
    return ((h&1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
}
 
double noise(double x, double y, double z)
{
    int X = (int)floor(x) & 255,
        Y = (int)floor(y) & 255,
        Z = (int)floor(z) & 255;
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);
    double u = fade(x),
           v = fade(y),
           w = fade(z);
    int A = p[X  ]+Y, AA = p[A]+Z, AB = p[A+1]+Z,
        B = p[X+1]+Y, BA = p[B]+Z, BB = p[B+1]+Z;
 
    return lerp(w, lerp(v, lerp(u, grad(p[AA  ], x  , y  , z   ),
                                   grad(p[BA  ], x-1, y  , z   )),
                           lerp(u, grad(p[AB  ], x  , y-1, z   ),
                                   grad(p[BB  ], x-1, y-1, z   ))),
                   lerp(v, lerp(u, grad(p[AA+1], x  , y  , z-1 ),
                                   grad(p[BA+1], x-1, y  , z-1 )),
                           lerp(u, grad(p[AB+1], x  , y-1, z-1 ),
                                   grad(p[BB+1], x-1, y-1, z-1 ))));
}
```

## 2. 对物体建模

可用 Grid 立方体格子形式建模

```cpp
struct Grid {
	// 密度指针：指向一个存储着所有体素voxel的密度的数组
    float *density;
    // 维度：沿着一个轴方向上的划分的voxel数量（voxel总数为d*d*d）
    size_t dimension = 128;
    // 世界坐标：用两个点表示一个立方体格子的世界坐标
    Point bounds[2] = { (-30,-30,-30), (30, 30, 30) };
};
```

下图这个大正方体为一个 Grid，每个小正方体为一个voxel，density中存储着每个voxel 的密度，通过对不同voxel 赋予不同的density，就可以显示出不同的物体形状

![[Pasted image 20230803113410.png|300]]

### 1. 渲染过程

渲染过程与上面[[#1. Procedure 过程|Procedure 过程]]类似，不过有一下几点改变了：

1. **Intersection 相交**：光线与Grid 相交的时候，就是与[[5.Ray Tracing光线追踪#3. 光线物体表面相交的加速（AABB包围盒方法）|AABB相交]]（可以得到$t_{enter}$ 和 $t_{exit}$）
2. **采样点的density密度**：原本的密度是通过Berlin Noise得到的，现在是通过采样点的**坐标**来得到对应 voxel 的 density（这些density是通过读取数据库里面的不同类型物体的density得到的）

```cpp
float evalDensity(const Grid* grid, const Point& p)
{
	// Grid的边长
    Vector gridSize = grid.bounds[1] - grid.bounds[0];
    // 当前点的占比
    Vector pLocal = (p - grid.bounds[0]) / gridSize;
    // 当前点的像素值（现在还是float）
    Vector pVoxel = pLocal * grid.baseResolution;

    int xi = static_cast<int>(std::floor(pVoxel.x));
    int yi = static_cast<int>(std::floor(pVoxel.y));
    int zi = static_cast<int>(std::floor(pVoxel.z));

    // nearest neighbor
    return grid->density[(zi * grid->resolution + yi) * grid->resolution + xi]; 
}
```

### 2. Trilinear三线性插值优化密度值

原本是直接得到当前坐标得像素值（nearest neighbor），现在可以用周围最近的八个像素点的坐标来进行 **trilinear 三线性插值**（其中可以将点坐标 p - 0.5，这样进行向下取整后得到的点坐标就是8个点中最小的点）

![[Pasted image 20230803114733.png]]

先对 x轴（红色）方向进行4次线性插值，再对 y轴（绿色）方向进行2次线性插值，最后对 z轴（蓝色）方向进行一次线性插值

```cpp
float result = 
    (1 - z0) * (  // blue
                (1 - y0) * (v000 * (1 - x0) + v100 * x0) + // green and red
                     y0  * (v010 * (1 - x0) + v110 * x0)   // green and red
                ) + 
          z0 * ( // blue
                (1 - y0) * (v001 * (1 - x0) + v101 * x0) + // green and red
                     y0  * (v011 * (1 - x0) + v111 * x0)); // green and red

// 也等同于下面，其中wx0 = 1-x0, wx1 = x0（y、z方向类似）
float result = 
    v000 * wx0 * wy0 * wz0 +
    v100 * wx1 * wy0 * wz0 +
    v010 * wx0 * wy1 * wz0 +
    v110 * wx1 * wy1 * wz0 +
    v001 * wx0 * wy0 * wz1 +
    v101 * wx1 * wy0 * wz1 +
    v011 * wx0 * wy1 * wz1 +
    v111 * wx1 * wy1 * wz1;
```

### 3. 密度存储优化

可以将密度值相同的voxel 合成一个大的voxel

![[Pasted image 20230803113020.png|400]] ![[Pasted image 20230803115331.png|200]]

---

## 3. Volume Rendering Equation

> 从理论方面研究上述用到的变量，以及惯例上常用的方程

#### 1. 参数

* $\sigma_a$： **absorption coefficient吸收系数**，也可以理解为probability density that light is absorbed per unit distance 光线单位距离被吸收的概率密度（单位为$m^{-1}$ / $cm^{-1}$等）
* $\sigma_s$：**scattering coefficient散射系数**，与$\sigma_a$ 类似，也可以理解为probability that a photon is being scattered by the volume per unit distance 光子单位距离被散射的概率（这里的概率应该就是上述的概率密度，因为都是 per unit distance）
* $\sigma_t = \sigma_a + \sigma_s$：**extinction / attenuation coefficient衰减系数**，因为 absorption 和 out-scattering 都会衰减光线，二者效果从结果上来说一样，所以可以放一起
* $T(s) = exp(-\int_i^s \sigma_t(s)\ ds)$：**transmittance透射率**，值越大表示投射/穿过的程度越大。其推导过程来自 Beer's law：$T(s) = \frac{L(s)}{L_i}$ $$\begin{aligned} dL(s) &= -\sigma_tL(s) \\ \frac{dL(s)}{ds} &= -\sigma_tL(s) \\ \frac{dL(s)}{L(s)} &= -\sigma_tds \\ \int_i^s\frac{1}{L(s)}dL(s) &= \int_i^s-\sigma_tds \\ lnL(s) &= -\int_i^s\sigma_tds+C \\ L(s) &= C'\cdot exp(-\int_i^s\sigma_tds) \end{aligned}$$当s = i（就是取初始值）时，$L(i) = L_i$，所以得到最终式子：$$L(s) = L_i\cdot exp(-\int_i^s\sigma_tds)$$
* $\tau = \int_i^o\sigma_t(s)\ ds$：**optical depth光学厚度**，是$\sigma_t$ 的对距离积分，得到光线被吸收的程度 / 概率，所以 $T(s)$ 也可以表达为 $T(s) = exp(-\tau)$
* $f_p(x, \omega, \omega') = f_p(x, \theta)$：**phase function**，描述了光线在位置x 处，在光线$\omega'$ 方向和相机 $\omega$ 方向夹角为 $\theta$ 时，光线的散射分布 / 程度
	1. $f_p(x,\omega,\omega') = \frac{1}{4\pi}$：Isotropic 各向同性，光线沿各个方向散射程度一样
	2. $f_p(x,\omega,\omega') = \frac{1}{4\pi}\cdot \frac{1-g^2}{(1+g^2-2g\cos\theta)^{\frac{3}{2}}}$：Anisotropic 各项异性，不同方向不同g值散射程度不同（g > 0时偏向 forward，g = 0时同各向同性，g < 0时偏向 backward）![[Pasted image 20230803153732.png]]

### 2. 求体渲染方程 Volume Rendering Equation

* 方法：建模成 光线沿直线穿过一堆粒子，计算光线从发射出到打到成像平面的辐射强度，就可以渲染出投影图像 

![[Pasted image 20230801213024.png|300]]

* 具体步骤：将光子与粒子发生作用的过程细分为四种类型——吸收、放射、外散射、内散射，就可以得到入射光和出射光之间的关系：$L_o - L_i = dL(x, \omega) = emission + inscattering - outscattering - absorption$

![[Pasted image 20230801213639.png]]

假设没有自发光（光源才有），那么可以得到：$$\frac{dL(x,\omega)}{dx} = -\sigma_tL(x,\omega) + \sigma_s\int_{S^2}f_p(x,\omega,\omega')\ L(\omega,\omega')\ d\omega'\tag{1}$$右式中的左边部分表示光线在传播过程中在**当前 $(x,\omega)$ 位置** absorption 和 out-sacttering 的部分（衰减），右边部分表示 in-scattering 部分（增强）。下面令$$L_s(x,\omega) = \int_{S^2}f_p(x,\omega,\omega')\ L(x,\omega')\ d\omega'\tag{2}$$于是方程 $(1)$ 可以化简为：$$\frac{dL(x,\omega)}{dx} = -\sigma_tL(x,\omega)\ + \ \sigma_sL_s(x,\omega)\tag{3}$$
* 解微分方程的补充：$$\begin{aligned} y'(x)+p(x)=q(x)\\y(x)=e^{-\int p(x)dx}(\int e^{-\int p(x)dx}\cdot q(x)\ dx\ +\ C) \end{aligned}$$
在方程 $(3)$ 中 $p(x)=\sigma_t$，$q(x)=\sigma_sL_s(x,\omega)$，于是可以得到：$$\begin{aligned} L(s)&=e^{-\int_0^s\sigma_t(s)ds}(\int_0^se^{\int_0^x\sigma_t(s)ds}\cdot[\sigma_s(s)\ L_s(s)]\ dx\ +\ C) \\ &= e^{-\int_0^s\sigma_t(s)ds}\cdot \int_0^se^{\int_0^x\sigma_t(s)ds}\cdot[\sigma_s(s)\ L_s(s)]\ dx\ +\ Ce^{-\int_0^s\sigma_t(s)ds} \\ &= \int_0^se^{-\int_0^s\sigma_t(s)ds}\cdot e^{\int_0^x\sigma_t(s)ds}\cdot[\sigma_s(s)\ L_s(s)]\ dx\ +\ Ce^{-\int_0^s\sigma_t(s)ds} \\ &=\int_0^se^{-\int_x^s\sigma_t(s)ds}\cdot[\sigma_s(s)\ L_s(s)]\ dx\ +\ Ce^{-\int_0^s\sigma_t(s)ds} \\ &=\int_0^se^{-\int_x^s\sigma_t(s)ds}\cdot[\sigma_s(s)\ L_s(s)]\ dx\ +\ L_0e^{-\int_0^s\sigma_t(s)ds} \end{aligned}\tag{4}$$ 这里的积分方向是从最远的交点（0）往相机（s）积分。第一个式子括号外面的e 的积分域在 0 ~ s 之间，表示从开始到最终位置；至于为什么括号里面的e 的积分域在 0 ~ x 之间，可以从结果来看：结果右边一项可以理解成**从 $L_0$ （背景光）到当前 / 最终位置s** 经过 absorption 和 out-scattering 后的值，左边可以理解成外界光源到达当前采样点x（这里是对x 积分，表示不同的采样点x，其color值为 $\sigma_s(s)\ L_s(s)$），然后经过从**采样点 x -> s 最终位置**的积分得到的 absorption 和 out-scattering 衰减后的值（可以看出，这个式子和上面我们代码实现的思想其实是一样的）

下图中也可以看出有两种光线合成得到最终的 $L(x,\omega)$，分别是**背景光直接衰减穿过**（$(4)$ 式的右边部分）和 **光源发出光经过衰减到达采样点 $x_t$ 后再衰减穿过到达人眼**（$(4)$ 式的左边部分）

![[Pasted image 20230803172256.png]]

因为 $T(s)=exp(-\int_0^s\sigma_t(x_t)dt)$ 表示的是从 0 到 s 处积分得到这一整段的透射率，积分起始位置用的都是0，所以我们可以**把上式切换一个方向——从相机往物体看**，那么起始位置就一直是进入的交点0，再带入上式可以得到：$$L(s)=\int_0^sT(t)\cdot[\sigma_s(x_t)\ L_s(x_t,\omega)]\ dt\ +\ L_0T(s)\tag{5}$$积分的左边原本$(4)$中指的是从采样点x 到相机s 的积分，现在转变成从相机0 到采样点t 的积分（积分中每次取一小段t 来得到采样点）二者是等价的；积分的右边$T(s)$一直是整段完整路径的积分，交换顺序结果一样。这个式子就是NeRF 中的Volume Rendering的式子（多了一个背景光 $L_0T(s)$）

![[Pasted image 20230801232832.png]]

### 3. 对照NeRF的公式来得到离散型的公式

因为积分在计算机中无法表达，需要**离散化**：将光路 \[0, s] 划分为 N 个等间距的区间 $[t_n, t_{n+1}]$ ，计算每个区间的辐射强度 $I(t_n\rightarrow t_{n+1})$ ，再累加起来就可以得到光线强度（N越大，越接近积分式）

假设每个区间内 $\sigma(t)$ 处处等于 $\sigma_n$ ，$C(t)$ 处处等于 $C_n$，则由 $(5)$ 可得（除去背景光项）：$$\begin{aligned}L(t_n \rightarrow t_{n+1}) &=\int_{t_n}^{t_{n+1}}T(t)\ \sigma_n\ C_n\ dt \\ &=\sigma_n\ C_n\ \int_{t_n}^{t_{n+1}}T(t)\ dt \end{aligned} \tag{6}$$
其中 $T(t)$ 可以分解成两段的乘积：$$\begin{aligned} T(t) &= exp(-\int_0^t \sigma(u)\ du) \\ &=exp(-[\int_o^{t_n}\ \sigma(u)\ du\ + \ \int_{t_n}^{t}\ \sigma(u)\ du]) \\ &=exp(-\int_0^{t_n}\ \sigma(u)\ du)\ exp(-\int_{t_n}^{t}\ \sigma(u)\ du) \\ &=T(0 \rightarrow t_n)\ T(t_n \rightarrow t) \end{aligned} \tag{7}$$
其中$T(0\rightarrow t_n)$ 无变量t，$T(t_n \rightarrow t)$ 有变量t，于是可以化简 $(6)$ 为：$$\begin{aligned} L(t_n \rightarrow t_{n+1}) &= \sigma_n\ C_n\ \int_{t_n}^{t_{n+1}}\ T(0 \rightarrow t_{n})\ T(t_n \rightarrow t)\ dt \\ &= \sigma_n\ C_n\ T(0 \rightarrow t_{n})\ \int_{t_n}^{t_{n+1}}\ T(t_n \rightarrow t)\ dt \\ &= \sigma_n\ C_n\ T(0 \rightarrow t_{n})\ \int_{t_n}^{t_{n+1}}\ exp(-\int_{t_n}^{t}\sigma_n\ du)\ dt \\ &= \sigma_n\ C_n\ T(0 \rightarrow t_{n})\ \int_{t_n}^{t_{n+1}}\ exp(-\sigma_n\int_{t_n}^{t}du)\ dt \\ &= \sigma_n\ C_n\ T(0 \rightarrow t_{n})\ \int_{t_n}^{t_{n+1}}\ exp(-\sigma_n(t-t_n))\ dt \\ &= \sigma_n\ C_n\ T(0 \rightarrow t_{n})\ \frac{exp(-\sigma_n(t-t_n))}{-\sigma_n}\mid_{t_n}^{t_{n+1}} \\ &= T(0\rightarrow t_n)\ C_n\ (1-exp(\sigma_n(t_{n+1}-t_n))) \end{aligned} \tag{8}$$
将所有区间累加起来简化公式 $(5)$ 得到：$$\begin{aligned} L(s) &= \int_0^sT(t)\ \sigma(t)\ C(t)\ dt + T(s)L_0 \\ &\approx \displaystyle \sum_{n=1}^{N}I(t_n \rightarrow t_{n+1})+T(s)L_0 \\ &= \displaystyle \sum_{n=1}^{N}T(0\rightarrow T_n)\ (1-exp(\sigma_n(t_{n+1}-t_n)))\ C_n\ +\ T(s)L_0 \end{aligned} \tag{9}$$
假设 $\delta_n = t_{n+1} - t_n$，其表示为一个小区间长度，且令 $T_n = T(0\rightarrow t_n)$，则：$$\begin{aligned} T_n&=exp(-\int_0^{t_n}\sigma(u)\ du) \\ &= exp(-\displaystyle\sum_{k=1}^{n-1}\int_{t_k}^{t_{k+1}}\sigma_k\ du) \\ &\approx exp(-\displaystyle\sum_{k=1}^{n-1}\sigma_k\ \delta_k) \end{aligned} \tag{10}$$
化简 $(9)$ 得到：$$\begin{equation} L(s) = \sum_{n=1}^N T_n\ (1-exp(-\sigma_n\delta_n))\ C_n + T(s)L_0 \end{equation} \tag{11}$$ 
公式 $(11)、(10)$ 与NeRF给出的离散化表示的体渲染方程类似（多了背景光）

![[Pasted image 20230802100630.png]]


## 4. 缺点与改进

* 缺点：
	1. 模拟光线在现实生活中如何表现的能力较差，我们只考虑了光子进行一次 interaction（absorption 或 scattering），而实际上会进行很多次scatterint（类比于直接光照效果没有直接光照+间接光照好）
	2. stochastic approach 在表现实际事物上更好


![[Pasted image 20230803200258.png]]

* 优化：通过模拟光子的 random walk 随机走动来模拟光子如何与物体粒子交互的（Monte Carlo particle transport(MCPT)）
