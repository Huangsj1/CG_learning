## 1. 求屏幕的三角形的任意点 P 的重心坐标 $(\alpha', \beta', \gamma')$

![[91df1d16f1abd047bc591d3b8fc9bb7.jpg|300]]

```cpp
auto [alpha, beta, gamma] = computeBarycentric2D(x + 0.5, y + 0.5, t.v);

// （alpha、beta、gamma）是三角形重心坐标
static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f* v) {
    float beta = ((y - v[0].y())*(v[2].x() - v[0].x()) - (x - v[0].x())*(v[2].y() - v[0].y())) / ((v[1].y() - v[0].y())*(v[2].x() - v[0].x()) - (v[1].x() - v[0].x())*(v[2].y() - v[0].y()));
    float gamma = ((y - v[0].y()) * (v[1].x() - v[0].x()) - (x - v[0].x()) * (v[1].y() - v[0].y())) / ((v[2].y() - v[0].y()) * (v[1].x() - v[0].x()) - (v[2].x() - v[0].x()) * (v[1].y() - v[0].y()));
    float alpha = 1 - beta - gamma;
    return { alpha, beta, gamma };
}

```

## 2. 根据[透视投影矫正](https://blog.csdn.net/motarookie/article/details/124284471)得到**原始**的 深度 z 和 属性 I

* 注意这里的深度测试用的是**原本的深度**，这样更准确，且不用求现在的深度；而且深度只是用来进行深度测试来判断是否替换的，对于绘图只用了 frame_buf\[width * height]

![[d4b16196198f5232777e26103a72a4c.jpg|300]] ![[3108f58186733377b0ce518ca9bb237.jpg|300]]

```cpp
void rst::rasterizer::rasterize_triangle(const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos) {
	// ...
	for (int x = min_x; x < max_x; x++) {
        for (int y = min_y; y < max_y; y++) {
            if (insideTriangle(x + 0.5, y + 0.5, t.v)) {
                // 1.得到重心坐标
                auto [alpha, beta, gamma] = computeBarycentric2D(x + 0.5, y + 0.5, t.v);
                // 2.得到之前的深度值 (公式2)
                float z_before = 1 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                
                // 3.深度测试(用原来的深度来进行测试)
                if (z_before > depth_buf[get_index(x, y)]) {
                    // 3.1.属性值通用公式得到各个属性的插值3)
                    auto normal_interpolated = z_before * (alpha * t.normal[0] / v[0].w() + beta * t.normal[1] / v[1].w() + gamma * t.normal[2] / v[2].w());
                    auto color_interpolated = z_before * (alpha * t.color[0] / v[0].w() + beta * t.color[1] / v[1].w() + gamma * t.color[2] / v[2].w());
                    auto textureCoord_interpolated = z_before * (alpha * t.tex_coords[0] / v[0].w() + beta * t.tex_coords[1] / v[1].w() + gamma * t.tex_coords[2] / v[2].w());
                    auto shadingCoords_interpolated = z_before * (alpha * view_pos[0] / v[0].w() + beta * view_pos[1] / v[1].w() + gamma * view_pos[2] / v[2].w());
                    
                    // 3.2.片段着色器(以面为单位来着色)
                    fragment_shader_payload payload(color_interpolated, normal_interpolated, textureCoord_interpolated, texture ? &*texture : nullptr);
                    payload.view_pos = shadingCoords_interpolated;
                    auto pixel_color = fragment_shader(payload);
                    Eigen::Vector2i point(x, y);
                    set_pixel(point, pixel_color);
                    depth_buf[get_index(x, y)] = z_before;
                }
            }
        }
	// ...
}
```

深度初始化要改成负无穷（因为现在的 depth_buf 用的是之前的深度值）

```cpp
void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        /*std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());*/
        std::fill(depth_buf.begin(), depth_buf.end(), -std::numeric_limits<float>::infinity());
    }
}
```

## 3. 片元着色器 fragment shader

知道了一个三角形的三个顶点的颜色、法线、纹理、原始坐标（经过mv变换），就可以通过2中的插值得到三角形内任意点P对应属性，根据属性就可以运用各种不同的 **shader** 来对**像素**进行渲染（赋予不同的color）

* `rasterize.fragment_shader` 是不同的 fragment shader **函数**，用于根据不同的属性来赋予color
* `fragment_shader_payload` 是包含shder所需要属性（三角形内当前**点P**的view_pos原本坐标、color颜色、normal法线、tex_coords纹理坐标、texture纹理图等）的**结构**，用于传给 fragment shader 函数使用

### 1. normal_fragment_shader 法线

根据法线的不同值赋予不同的color

```cpp
Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}
```

![[normal.png|300]]

### 2. phong_fragment_shader 光照

使用了 [[3. Shading(shading-pipeline-texture)#一、简单着色模型——Blinn-Phong Reflectance Model|Blinn-Phong 着色模型]]

1. 环境光 $ambient\_light = k_a*I$

![[Pasted image 20230718201051.png|200]]

2. 漫反射 $diffuse\_light = k_d*(\frac{I}{r^2})*max(0,n\cdot l)$ 
	* 这里的 $k_d$ 与当前点的 color 有关（毕竟只用显示出整体样子，颜色无所谓）

![[Pasted image 20230718203400.png|200]]

3. 反射 $specular\_light=k_s*(\frac{I}{r^2})*max(0,n\cdot h)$

![[Pasted image 20230718203620.png|220]]

4. 总和：环境光 + 漫反射 + 反射
	* 左边用的是**原点**作为视点（因为point已经经过了视图mv变换）：视线更靠近物体，所以发亮的范围少一点（角度变化大）
	* 右边用的是**eye_pos**坐标作为视点（二者结果差不多）：视线离物体远一点，发亮的范围多一点（角度变化小）

![[Pasted image 20230718201842.png|300]] ![[Pasted image 20230718201946.png|300]]

反射光的 p 从小 -> 大

![[Pasted image 20230718203909.png|150]] ![[Pasted image 20230718203933.png|150]] ![[Pasted image 20230718204015.png|150]] ![[Pasted image 20230718201842.png|300|150]]

```cpp
// 2.光照 phong shader
Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    // ka环境光系数ambient；kd漫反射光系数diffuse；ks反射光系数specular
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    // 两个点光源（位置，强度）
    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    // 这里的坐标和相机初始位置一样
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    // 这里计算的是每一个光源到像素点的最终强度（加和到result_color中）
    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.

        // 先计算得到法向量n、光照向量l、视觉向量v(化为单位向量减少计算量)
        Eigen::Vector3f normal_vector = normal.normalized();
        Eigen::Vector3f light_vector = (light.position - point).normalized();
        // 这里已经经过相机平移了，是否还需要移动视角？？
        Eigen::Vector3f view_vector = (Eigen::Vector3f(0, 0, 0) - point).normalized();
        /*Eigen::Vector3f view_vector = (eye_pos - point).normalized();*/
        Eigen::Vector3f half_vector = (view_vector + light_vector).normalized();

        // 1.环境光(假设是原本值)
        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);
        
        // 2.漫反射
        float r_square = (light.position - point).squaredNorm();
        Eigen::Vector3f diffuse = kd.cwiseProduct(light.intensity / r_square * std::max(0.f, normal_vector.dot(light_vector)));
    
        // 3.反射
        Eigen::Vector3f specular = ks.cwiseProduct(light.intensity / r_square * std::pow(std::max(0.f, normal_vector.dot(half_vector)), 300));

        result_color += ambient + diffuse + specular;
    }

    // 由光强 -> 颜色
    return result_color * 255.f;
}
```

### 3. texture_fragment_shader 纹理

已知三角形三个顶点的纹理坐标，通过插值就可直接得到任一点P的texture_coords，然后根据纹理坐标uv在纹理图片中获取颜色return_color，将该颜色来赋予 $k_d$ ，使得漫反射得到的结果颜色与纹理颜色有关

```cpp
// 3.纹理shader
Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        return_color = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    // 这里的漫反射系数取了与纹理颜色texture_color有关的值
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        // 先计算得到法向量n、光照向量l、视觉向量v(化为单位向量减少计算量)
        Eigen::Vector3f normal_vector = normal.normalized();
        Eigen::Vector3f light_vector = (light.position - point).normalized();
        Eigen::Vector3f view_vector = (Eigen::Vector3f(0, 0, 0) - point).normalized();
        Eigen::Vector3f half_vector = (view_vector + light_vector).normalized();

        // 1.环境光(假设是原本值)
        Eigen::Vector3f ambient = ka.cwiseProduct(amb_light_intensity);

        // 2.漫反射
        float r_square = (light.position - point).squaredNorm();
        Eigen::Vector3f diffuse = kd.cwiseProduct(light.intensity / r_square * std::max(0.f, normal_vector.dot(light_vector)));

        // 3.反射
        Eigen::Vector3f specular = ks.cwiseProduct(light.intensity / r_square * std::pow(std::max(0.f, normal_vector.dot(half_vector)), p));

        result_color += ambient + diffuse + specular;
    }

    return result_color * 255.f;
}
```

* 注意：得到的uv坐标还要限制范围（uv坐标的值是在 \[ 0, 1] 之间，是比例），小于0置0，大于1置1；最后再放到到 \[width] \[height] 范围

```cpp
Eigen::Vector3f getColor(float u, float v)
{
	// 限制坐标范围
	if (u < 0) u = 0;
	if (u > 1) u = 1;
	if (v < 0) v = 0;
	if (v > 1) v = 1;
	// 放大到width、height
	auto u_img = u * width;
	auto v_img = (1 - v) * height;
	auto color = image_data.at<cv::Vec3b>(v_img, u_img);
	return Eigen::Vector3f(color[0], color[1], color[2]);
}
```

![[texture.png|300]]

### 4. bump_fragment_shader 凹凸贴图

`texture` 中存储了纹理的 width、height，其 `getColor()` 得到的 Vector3f 经过 `.norm()` 后得到的是**高度值**，也就是说凹凸贴图存储的是**高度信息**，在不改变模型的情况下通过不同的高度来**改变法线**，使得渲染出来的图形更逼真（相对于直接渲染精细的模型时间更短，空间换时间）

由[[3. Shading(shading-pipeline-texture)#1. 凹凸贴图 Bump Mapping|之前的推导]]可以得到：法线 $(0, 0, 1)$ 对应的经过凹凸贴图后的法向量为 $(-dU, -dV, 1)$ ，而实际上的法向量 `normal` 要先经过 **TBN 矩阵**的变换变成 $(0,0,1)$ ，再乘以 **$(-dU, -dV, 1)$** 才能正确的到对应的经过凹凸贴图后的法向量

```cpp
// 4.bump shader 凹凸贴图：通过高度变化来使得表面发生微小变化
Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    // 系数c1、c2
    float kh = 0.2, kn = 0.1;

    // TODO: Implement bump mapping here
    // 1.先求TBN矩阵（用来将切线空间和世界坐标空间联系起来）
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    float x = normal.x();
    float y = normal.y();
    float z = normal.z();
    Eigen::Vector3f t, b;
    t << x * y / std::sqrt(x * x + z * z), std::sqrt(x * x + z * z), z * y / std::sqrt(x * x + z * z);
    b = normal.cross(t);
    Eigen::Matrix3f TBN;
    TBN << t.x(), b.x(), normal.x(),
        t.y(), b.y(), normal.y(),
        t.z(), b.z(), normal.z();

    // 2.再求法线为(0,0,1)对应的贴图的法线(-du,-dv,1)
    // uv就是纹理坐标，范围在[0,1]之间
    float u = payload.tex_coords.x();
    float v = payload.tex_coords.y();
    float w = payload.texture->width;
    float h = payload.texture->height;
    // 凹凸贴图中的getColor存储的是高度信息，所以要.norm()
    // h()指的是高度函数，h(u+1)实际上是往u方向走一个单位，而这里的u是比例，所以应该写成h(u+1/width, v)
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    float dU = kh * kn * (payload.texture->getColor(u + 1 / w, v).norm() - payload.texture->getColor(u, v).norm());
    float dV = kh * kn * (payload.texture->getColor(u, v + 1 / h).norm() - payload.texture->getColor(u, v).norm());

    // ln求的是(0,0,1)坐标对应的变换后的法线，但是初始的normal值不是(0,0,1)，所以要经过TBN转换后才行
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)
    Eigen::Vector3f ln;
    ln << -dU, -dV, 1;
    normal = (TBN * ln).normalized();

    Eigen::Vector3f result_color = { 0, 0, 0 };
    result_color = normal;

    return result_color * 255.f;
}
```

可以看到，在原本 normal shader 渲染出来很光滑的图形上，出现了各种凹凸变化（这实际上只是法线的变化，点实际位置没有发生变化） （且这里只是在normal shader之上改进，没有加入光照phong 和纹理贴图texture）

![[Pasted image 20230719111226.png|300]]

### 5. displacement_fragment_shader 位移贴图

在 bump shader的基础上（只改变了法线，没有像bump shader一样用法线来作为颜色渲染），**改变了顶点 point 的高度**；并加上了光照phong shader

```cpp
// Position p = p + kn * n * h(u,v)
point += kn * normal * payload.texture->getColor(u, v).norm();
```

![[Pasted image 20230719113840.png|300]]


## 4. 双线性插值

![[d12030076ae9d38121522d676195a4c.jpg|300]]

```cpp
// 双线性插值
Eigen::Vector3f getColor_binary(float u, float v)
{
	// 限制坐标范围
	if (u < 0) u = 0;
	if (u > 1) u = 1;
	if (v < 0) v = 0;
	if (v > 1) v = 1;
	// 这里u为向右方向；v可能是因为坐标是从上往下取的，而纹理坐标v是从下往上
	float p_x = u * width;
	float p_y = (1 - v) * height;
	// 1.根据点P的位置找出最近的四个整数点
	int x0, x1, y0, y1;
	// (1.如果P点位于当前像素中心左边，就取x0为左边，x1为当前；如果P靠右边，就取x0为当前，x1为右边
	if (p_x - int(p_x) < 0.5) {
		x0 = std::max(0, int(p_x) - 1);
		x1 = p_x;
	}
	else {
		x0 = p_x;
		x1 = std::min(width, int(p_x) + 1);
	}
	// (2.如果P点位于当前像素中心下面，就取y0为下面，y1为当前；如果P靠上面，就取y0为当前，y1为上面；
	if (p_y - int(p_y) < 0.5) {
		y0 = std::max(0, int(p_y) - 1);
		y1 = p_y;
	}
	else {
		y0 = p_y;
		y1 = std::min(height, int(p_y) + 1);
	}

	// 如果在边界的像素，下面双线性插值就会除0错误，所以在边界直接返回单个纹理元素值即可
	if (x0 == x1 || y0 == y1) {
		return getColor(u, v);
	}

	// 2.进行双线性插值（先两次x轴的，再一次y轴的）
	// 注意这里的x、y参数顺序
	auto q1 = image_data.at<cv::Vec3b>(y0, x0);
	auto q2 = image_data.at<cv::Vec3b>(y0, x1);
	auto q3 = image_data.at<cv::Vec3b>(y1, x0);
	auto q4 = image_data.at<cv::Vec3b>(y1, x1);

	Eigen::Vector3f color_q1;
	Eigen::Vector3f color_q2;
	Eigen::Vector3f color_q3;
	Eigen::Vector3f color_q4;

	color_q1 << q1[0], q1[1], q1[2];
	color_q2 << q2[0], q2[1], q2[2];
	color_q3 << q3[0], q3[1], q3[2];
	color_q4 << q4[0], q4[1], q4[2];

	auto color_r1 = (x1 - p_x) * color_q1 / (x1 - x0) + (p_x - x0) * color_q2 / (x1 - x0);
	auto color_r2 = (x1 - p_x) * color_q3 / (x1 - x0) + (p_x - x0) * color_q4 / (x1 - x0);
	auto color_p = (y1 - p_y) * color_r1 / (y1 - y0) + (p_y - y0) * color_r2 / (y1 - y0);

	return color_p;
}
```

![[Pasted image 20230719162737.png|300]]

通过放大对比可以看到，左图（直接 `getColor`）得到的有较多一块一块的相同颜色的像素，锯齿较明显；右图（双线性插值 `getColor_binary`）得到的像素颜色更多渐变，可以达到**模糊抗锯齿**的作用

![[Pasted image 20230719163029.png]]

# 渲染管线（流程）

![[0fa57e0f55c8622734785c246d28226.jpg]]

## 1. 点的输入

获取**所有顶点属性值**（坐标、法线、纹理坐标；颜色是通过光线、纹理等来得到的），同时也存储进三角形中

```cpp
int main(int argc, const char** argv) {
	//...
	// Load .obj File
    bool loadout = Loader.LoadFile("models/spot/spot_triangulated_good.obj");
    // 加载得到每个三角形的顶点的 坐标vertex、法线normal、纹理坐标texcoord
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
                t->setVertex(...);
                t->setNormal(...);
                t->setTexCoord(...);
            }
            TriangleList.push_back(t);
        }
    }
    // ...
}
```

## 2. 三角形内的顶点处理

将所有三角形传入 `r.draw()` 中，对所有顶点进行**mvp变换**和**缩放适应屏幕**大小

```cpp
void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList) {}
```

## 3. 光栅化 + 片元处理

对于每一个三角形，都要通过**深度测试**判断屏幕中的像素是否在该三角形内，经过测试的像素再**插值得到各种属性**（法线、纹理坐标、原始坐标、固定的颜色等），将这些属性放到 `payload` 中传给**片元着色器着色**（根据不同的着色方法来返回对应的color），并将这些color值存储到屏幕 frame_buf 中

```cpp
void rst::rasterizer::rasterize_triangle(const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos) {}

// 1.法线 normal shader
Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload) {}

// 2.光照 phong shader
Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload) {}

// 3.纹理shader
Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload) {}

// 4.bump shader 凹凸贴图：通过高度变化来使得表面发生微小变化
Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload) {}

// 5.displacement shader 位移贴图：改变顶点的位置来更加逼真
Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload) {}
```

## 4. 屏幕颜色输出

通过 `cv::` 下的一堆函数来将存储的屏幕颜色 frame_buf \[width * height] 进行**颜色显示**

```cpp
int main(int argc, const char** argv) {
	// ...
	// r.draw(TriangleList);
	cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
	image.convertTo(image, CV_8UC3, 1.0f);
	cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
	
	cv::imshow("image", image);
	cv::imwrite(filename, image);
	// ...
}
```