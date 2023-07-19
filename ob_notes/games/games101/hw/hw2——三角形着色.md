## 1. 绘制投影后的图

1. 对二维平面计算得到所有三角形的**包围盒**（二维中能包住三角形的最小矩形）
2. 在包围盒中遍历所有像素点，判断**是否在三角形内**（叉乘根据符号是否都相同判断）
3. 如果在三角形内部，**插值得到深度值、颜色**
	* 这里求出的 $\alpha,\beta,\gamma$ 是**屏幕空间**中三角形重心坐标（通过重心坐标可以表示三角形内任意点的属性值，如深度、颜色、纹理等），但是这只是屏幕空间中的，需要经过矫正才能得到原本三维真实属性值 $z_p = \frac{1}{\frac{\alpha}{z_A} + \frac{\beta}{z_B} + \frac{\gamma}{z_C}}$ $I_p = z_p({\alpha \frac{I_A}{z_A} + \beta\frac{I_B}{z_B} + \gamma\frac{I_C}{z_C}})$ （这其中的 z 是原本的z值，可由点左乘透视投影矩阵后的w得到）
	* 由于[[1. vector-matrix-viewing transformation#^pp-23-7-14|透视投影]]是**非线性**的，所以不能直接通过屏幕的三角形的顶点来插值得到其他点属性，还需要加上**修正**——[矫正透视投影插值](https://zhuanlan.zhihu.com/p/400257532)（屏幕上的三角形内的点的插值信息和圆三角形内的点的插值信息不同）
4. 如果靠近相机，就**更新深度缓冲区**depth buffer

### 显示的原理

![[710033534b899e7e543c611e42da2f9.jpg|400]]

在mvp矩阵之后还要将**z轴反转**，这样投影后的深度就是越大越远，且投影后的深度由负值变成正值，并可在最后宽高深移动和扩大的时候正确处理

```cpp
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
    float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.

    // 一、压缩成长方体（这里推到的时候直接用的是坐标）
    Eigen::Matrix4f perspective;
    perspective << zNear, 0, 0, 0,
        0, zNear, 0, 0,
        0, 0, zNear + zFar, -zNear * zFar,
        0, 0, 1, 0;

    // 二、再正交缩放到[-1,1]
    eye_fov = eye_fov * MY_PI / 180;
    float top = (abs(zNear) * tan(eye_fov / 2));
    float bottom = -top;
    float right = top * aspect_ratio;
    float left = -right;
    // 1.移动中心到z轴(因为这里给出的是fov和aspect，锥体中心已经在z轴上)
    Eigen::Matrix4f move;
    move << 1, 0, 0, -(right + left) / 2,
        0, 1, 0, -(top + bottom) / 2,
        0, 0, 1, -(zNear + zFar) / 2,
        0, 0, 0, 1;
    // 2.缩放到[-1,1]
    Eigen::Matrix4f scale;
    scale << 2 / (right - left), 0, 0, 0,
        0, 2 / (top - bottom), 0, 0,
        0, 0, 2 / (zNear - zFar), 0,
        0, 0, 0, 1;

    // 3.由于z-buffer越小越远，所以还要乘以一个镜像矩阵让z轴旋转180°
    Eigen::Matrix4f mirror;
    mirror << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1;

    projection = mirror * scale * move * perspective * projection;

    return projection;
}
```

### 透视投影矫正

* 注意：这里的插值得到深度用的是现在的深度，应该为用投影之前的深度，详细见[[hw3——三维模型shading]]里有修改

```cpp
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();

    // 1. 先找到平面中包围三角形的最小包围盒（矩形）
    float min_x = width;
    float max_x = 0;
    float min_y = height;
    float max_y = 0;

    for (int i = 0; i < 3; i++) {
        min_x = std::min(min_x, v[i].x());
        max_x = std::max(max_x, v[i].x());
        min_y = std::min(min_y, v[i].y());
        max_y = std::max(max_y, v[i].y());
    }

    // 2. 在包围盒中判断每一个像素点的中心坐标是否在三角形内
    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {
            // 如果像素点中心在三角形内
            if (insideTriangle(x + 0.5, y + 0.5, t.v)) {
                // 3.先插值得到该点的深度
                auto[alpha, beta, gamma] = computeBarycentric2D(x + 0.5, y + 0.5, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;

                // 4.再判断是否更靠近，若更靠近就修改color和修改buf的深度
                int index = get_index(x, y);
                if (depth_buf[index] > z_interpolated) {
                    Eigen::Vector3f point;
                    point << x, y, z_interpolated;
                    set_pixel(point, t.getColor());
                    depth_buf[index] = z_interpolated;
                }
            }
        }
    }
}
```

```cpp
// 判断点是否在三角形内
static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    int same = -1;

    for (int i = 0; i < 3; i++) {
        Eigen::Vector3f edge = _v[(i + 1) % 3] - _v[i];
        Eigen::Vector3f p = Eigen::Vector3f( x, y, 0 ) - _v[i];

        // 叉乘得到z的正负
        float cp = edge.cross(p).z();
        // 点在线上就继续判断
        if (cp == 0) continue;

        // 出现不同方向/符号的z就不在三角形内
        int flag = cp > 0 ? 1 : 0;
        if (same == -1) same = flag;
        if (same != flag) return false;
    }
    return true;
}
```

## 2. Anti-Aliasing 反走样方法

### 1. SSAA

* **SSAA**： 将一个像素点pixel分成多个采样点sample，对**所有采样点**判断是否在三角形内，对于在三角形内的采样点，每个都要**单独插值计算深度**，深度更浅的采样点就计算得到**它的color值**并存储（color计算**多次**，每个符合的采样点都要），并且保存**所有**采样点的深度、颜色信息；最后再取**平均值**作为该像素点的颜色值

1. 为 rasterize 结构体中添加成员变量来记录 **采样点** 的 **深度**和**颜色**（这里将采样点当作像素点来操作，最后输出时才取平均给像素点）

```cpp
// SSAA的frame_buf和depth_buf
class rasterizer {
	//...
    std::vector<std::vector<Eigen::Vector3f>> frame_buf_4xSSAA;
    std::vector<std::vector<float>> depth_buf_4xSSAA;
    //...
};
```

2. 光栅化三角形的时候对**采样点**判断深度和着色

```cpp
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
	// ...
    // 2. 在包围盒中判断每一个像素点的中心坐标是否在三角形内
    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {
            if (SSAA) {
                int index = get_index(x, y);
                // 遍历所有采样点
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        float sx = x + 0.25 + j * 0.5;
                        float sy = y + 0.25 + i * 0.5;
                        // 如果采样点在三角形内
                        if (insideTriangle(sx, sy, t.v)) {
                            // 插值得到深度
                            auto [alpha, beta, gamma] = computeBarycentric2D(sx, sy, t.v);
                            float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                            float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                            z_interpolated *= w_reciprocal;

                            // 判断是否更加靠近，若靠近就替换
                            if (depth_buf_4xSSAA[index][i * 2 + j] > z_interpolated) {
                                Eigen::Vector3f point;
                                point << sx, sy, z_interpolated;
                                depth_buf_4xSSAA[index][i * 2 + j] = z_interpolated;
                                frame_buf_4xSSAA[index][i * 2 + j] = t.getColor();
                            }
                        }
                    }
                }
            }
    // ...
}
```

3. 所有三角形光栅化结束后，通过对像素的所有采样点取**平均**得到像素的颜色

```cpp
void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
	// ...
	rasterize_triangle(t);
	
	// 通过对像素的采样点的平均得到像素的着色值
	if (SSAA) {
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				int index = get_index(x, y);
				Eigen::Vector3f color(0, 0, 0);
				for (int i = 0; i < 4; i++) {
					color += frame_buf_4xSSAA[index][i];
				}
				// 这里已经对所有三角形进行深度判断完了，显示颜色时不再需要用到深度信息，所以z值可以随意取
				// 不要忘了取平均值
				set_pixel(Eigen::Vector3f(x, y, 0), color / 4);
			}
		}
	}
}
```

### 2. MSAA

* **MSAA**：将一个像素点pixel分成多个采样点sample，对**所有采样点**判断是否在三角形内，对于在三角形内的采样点，每个都要**单独插值计算深度**，深度更浅的采样点的color就**用像素中心点的color**来存储（color计算**一次**）；需要保存到**所有**采样点的深度、颜色信息；最后也取**平均值**作为该像素点的颜色值

1， 3步骤同SSAA

2. 光栅化三角形的时候，对在三角形内部的采样点计算深度，满足深度测试的采样点就用**像素中心点的color**值来copy

```cpp
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
	// ...
    // 2. 在包围盒中判断每一个像素点的中心坐标是否在三角形内
    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {
	        // SSAA...
			else if (optimal == MSAA) {
					// 计算当前像素点的采样点在三角形内部的个数和掩码
					int tot = 0;
					int mask[4] = { 0, 0, 0, 0 };
					// 遍历所有采样点
					for (int i = 0; i < 2; i++) {
						for (int j = 0; j < 2; j++) {
							if (insideTriangle(x + 0.25 + j * 0.5, y + 0.25 + i * 0.5, t.v)) {
								++tot;
								mask[i * 2 + j] = 1;
							}
						}
					}
					// 如果有采样点在三角形内
					if (tot > 0) {
						int index = get_index(x, y);
						// 先插值得到像素中心点的颜色（与SSAA唯一不同的地方就是这里，MSAA用的是像素中心点的颜色，SSAA用的是采样点得到的颜色）
						Eigen::Vector3f color = t.getColor();
						// 将中心点的深度和颜色赋给所有采样点（只有深度更小才更新采样点的颜色）
						for (int i = 0; i < 2; ++i) {
							for (int j = 0; j < 2; ++j) {
								if (mask[i * 2 + j] != 0) {
									// 对所有在三角形内的采样点进行深度测试
									float sx = x + 0.25 + j * 0.5;
									float sy = y + 0.25 + j * 0.5;
									auto [alpha, beta, gamma] = computeBarycentric2D(sx, sy, t.v);
									float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
									float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
									z_interpolated *= w_reciprocal;
			
									// 判断是否更加靠近，若靠近就替换
									if (depth_buf_4xSSAA[index][i * 2 + j] > z_interpolated) {
										depth_buf_4xSSAA[index][i * 2 + j] = z_interpolated;
										// MSAA用的是像素中心点的颜色
										frame_buf_4xSSAA[index][i * 2 + j] = color;
									}
								}
							}
						}
					}
				}
			// 普通的
	// ...
}
```