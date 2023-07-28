# 流程

1. `main.cpp` 中先**创建场景**，包括放两个球、两个大三角形组成的地板，同时添加两个点光源，然后开始渲染
2. `Renderer::Render()`中通过给定 fov、aspect_ratio和相机位置，然后**遍历Render space的所有像素**（所有frame_buffer），再得到像素点的**world space的近平面位置**后，从相机处**沿着像素方向射出光线**来得到该像素的color
3. `castRay()`中根据入射光的起点和方向，向场景中**射出光线**来求是否与场景中的物体有碰撞，得到距离入射光最近的碰撞物体，根据物体**材质**来选择得到color的方式
	1. 反射 + 折射材质：得到反射光和折射光来作为入射光递归进行`castRay()`，当递归深度达到一定就退出
	2. 完美反射材质：将反射光作为入射光递归进行`castRay()`
	3. 漫反射材质：通过Blinn Phong模型将漫反射+反射作为color
4. `trace()`根据入射光的起点和方向，对于所有物体都**计算入射光是否于其相交**（这里包括球Sphere和三角形网格Triangle Mesh），记录相交点的信息

## 2. 将像素在Render space的坐标转换为World space的坐标

从屏幕位置（x，y）逆推回到 world space 中的近平面的坐标（x'，y'）

![[80e34dc91a675e4c2d3ffecf46b7b9c.jpg|300]]![[a582a0e9689aa83f0fb201f7b8cc011.jpg|300]]

```cpp
// 遍历所有屏幕上的像素得到其在world space的近平面的x、y位置来得到光线向量
void Renderer::Render(const Scene& scene)
{
    std::vector<Vector3f> framebuffer(scene.width * scene.height);

    // scale缩放 = tan(fov / 2)；aspect_ratio宽高比 = 宽 / 高
    float scale = std::tan(deg2rad(scene.fov * 0.5f));      // tan45°= 1
    float imageAspectRatio = scene.width / (float)scene.height;

    // Use this variable as the eye position to start your rays.
    Vector3f eye_pos(0);
    int m = 0;

    for (int j = 0; j < scene.height; ++j)
    {
        for (int i = 0; i < scene.width; ++i)
        {
            // generate primary ray direction
            float x = (2 * ((i + 0.5) / scene.width) - 1) * scale * imageAspectRatio;
            float y = (2 * (((scene.height - 1 - j) + 0.5) / scene.height) - 1) * scale;

            // 这里的-1也说明了近平面 n = -1
            Vector3f dir = Vector3f(x, y, -1); 
            // 标准化
            dir = normalize(dir);
            framebuffer[m++] = castRay(eye_pos, dir, scene, 0);
        }
        UpdateProgress(j / (float)scene.height);
    }

    // save framebuffer to file
    FILE* fp = fopen("binary.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (char)(255 * clamp(0, 1, framebuffer[i].x));
        color[1] = (char)(255 * clamp(0, 1, framebuffer[i].y));
        color[2] = (char)(255 * clamp(0, 1, framebuffer[i].z));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);    
}

```

## 3. 反射光 和 折射光向量 以及 菲涅尔项 的计算

![[d8a8802786cd8044df3764185269b1a.jpg|400]]

### 1. [[6.Materials and Appearances 材质与外观#1. 完美镜面反射 Perfect Specular reflection|反射光向量]]（这里 I 为从光源到表面）
```cpp
// Compute reflection direction
Vector3f reflect(const Vector3f &I, const Vector3f &N)
{
    return I - 2 * dotProduct(I, N) * N;
}
```

### 2. [[6.Materials and Appearances 材质与外观#2. 镜面折射 Specular refraction|折射光向量]]（这里 I 为从光源到表面）

```cpp
// 得到折射向量 T = ηI + (ηc1 - c2)N
// 其中η为eta，c1为cosi，c2为sqetf(k)
Vector3f refract(const Vector3f &I, const Vector3f &N, const float &ior)
{
    float cosi = clamp(-1, 1, dotProduct(I, N));
    // etai为空气中(外)的折射率η1，etat为其他介质中(内)的折射率η2
    float etai = 1, etat = ior;
    Vector3f n = N;
    // cosi < 0表示由外向内；cosi > 0表示由内向外，需要将N反向且交换etai和etat
    if (cosi < 0) { cosi = -cosi; } else { std::swap(etai, etat); n= -N; }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    // 如果k < 0 表示发生全反射（θ1 < θ2），没有折射
    return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
}
```

### 3. [[6.Materials and Appearances 材质与外观#3. 菲涅尔项 Fresnel Reflection / Term（反射率）|菲涅尔项]]（反射率）

```cpp
// 计算菲涅尔反射率
float fresnel(const Vector3f &I, const Vector3f &N, const float &ior)
{
    float cosi = clamp(-1, 1, dotProduct(I, N));
    float etai = 1, etat = ior;
    if (cosi > 0) {  std::swap(etai, etat); }
    // Compute sini using Snell's law
    float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1) {
        return 1;
    }
    else {
        float cost = sqrtf(std::max(0.f, 1 - sint * sint));
        cosi = fabsf(cosi);
        float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        return (Rs * Rs + Rp * Rp) / 2;
    }
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;
}
```

### 4. 反射/折射后的起点位置的偏移

由于**浮点数计算精度**问题，如果直接将交点值作为反射/折射后的光线起点就会出问题：反射/入射光线本应该从物体表面出发，但由于计算的舍入精度导致**偏内/偏外**，从而使得再次碰到交点附近的点而出问题（如下图）

![[Pasted image 20230725110629.png]]

![[204f168592416d396fbb37515d507fa.jpg|400]]

1. 若反射/折射光线与法线N同向就往N偏移
2. 与N反向就往-N偏移

```cpp
Vector3f castRay(const Vector3f &orig, const Vector3f &dir, const Scene& scene,int depth)
{
    // ...
	// 1.透明材质：反射+折射
	case REFLECTION_AND_REFRACTION:
	{
		// ...
		Vector3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ?
	hitPoint - N * scene.epsilon : hitPoint + N * scene.epsilon;
		Vector3f refractionRayOrig = (dotProduct(refractionDirection, N) < 0) ?
	hitPoint - N * scene.epsilon : hitPoint + N * scene.epsilon;
		// ...
		break;
	}
	// 2.镜子等材质：纯反射
	case REFLECTION:
	{
		// ...
		Vector3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ?
	hitPoint + N * scene.epsilon : hitPoint - N * scene.epsilon;
		hitColor = castRay(reflectionRayOrig, reflectionDirection, scene, depth + 1) * kr;
		// ...
		break;
	}
	// 3.默认材质：用Blinn Phong模型
	default:
	{
		// ...
		Vector3f shadowPointOrig = (dotProduct(dir, N) < 0) ?
	hitPoint + N * scene.epsilon : hitPoint - N * scene.epsilon;
		// ...
		break;
	}
	// ...
}
```

![[Pasted image 20230725111101.png]]

## 4. 光线与球和三角形相交 intersect

### 1. [[5.Ray Tracing光线追踪#1. 与隐式表面相交|与球相交]]（将光线向量带入球的方程）

```cpp
// orig为起点；dir为方向向量；tnear为交点的最短距离（进入距离）
// (o + td - c)^2 - r^2 = 0
// 有交点就返回true，同时传值给tnear
bool intersect(const Vector3f& orig, const Vector3f& dir, float& tnear, uint32_t&, Vector2f&) const override
{
	// analytic solution
	Vector3f L = orig - center;
	float a = dotProduct(dir, dir);
	float b = 2 * dotProduct(dir, L);
	float c = dotProduct(L, L) - radius2;
	float t0, t1;
	if (!solveQuadratic(a, b, c, t0, t1))
		return false;
	if (t0 < 0)
		t0 = t1;
	if (t0 < 0)
		return false;
	tnear = t0;

	return true;
}
```

### 2. [[5.Ray Tracing光线追踪#Möller Trumbore Algorithm 计算交点并判断是否在三角形内|与三角形相交]]（将光线向量带入三角形重心坐标计算）

```cpp
// 光线与单个三角形的交点 o + td = (1 - u - v)*v0 + u*v1 + v*v2
bool rayTriangleIntersect(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2, const Vector3f& orig,
                          const Vector3f& dir, float& tnear, float& u, float& v)
{
    Vector3f E1, E2, S, S1, S2;
    E1 = v1 - v0;
    E2 = v2 - v0;
    S = orig - v0;
    S1 = crossProduct(dir, E2);
    S2 = crossProduct(S, E1);
    float tmp = dotProduct(S1, E1);
    tnear = dotProduct(S2, E2) / tmp;
    u = dotProduct(S1, S) / tmp;
    v = dotProduct(S2, dir) / tmp;

    if (tnear > 0 && (1 - u - v >= 0 && 1 - u - v <= 1 && u >= 0 && u <= 1 && v >= 0 && v <= 1))
        return true;

    return false;
}
```