## 1. 初始化阶段

```cpp
// 创建了700*700个像素大小的背景
rst::rasterizer r(700, 700);

// 相机位置坐标是（0，0，5）位于z轴上，其g、t方向都是符合了的
Eigen::Vector3f eye_pos = {0, 0, 5};

// 三角形三个点的坐标位置
std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};
// 三个点的下标
std::vector<Eigen::Vector3i> ind{{0, 1, 2}};
```

## 2. 核心部分（对三角形进行视图变换 + 绘制三角形）

### 1. 视图变换

1. model transformation：模型变换（搭好场景——只是模型移动）
2. view transformation：视图变换（找好相机角度——相机和模型一起移动）
3. projection transformation：投影变换（将三维空间投影到二维空间）

```cpp
// 1. 模型变换：只用对三角形进行坐标变换
// 1）只用绕z轴旋转angle度
Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    
    // 角度转为弧度制
    rotation_angle = rotation_angle * MY_PI / 180;
    Eigen::Matrix4f rotation;
    // 绕z轴旋转的矩阵
    rotation << cos(rotation_angle), -sin(rotation_angle), 0, 0,
        sin(rotation_angle), cos(rotation_angle), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    model = rotation * model;
    
    return model;
}
// 2）绕任意轴旋转angle度
Eigen::Matrix4f get_rotation(Vector3f axis, float angle)
{
    angle = angle * MY_PI / 180;
    Eigen::Matrix4f rotation;
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f N;
    N << 0, -axis[2], axis[1],
        axis[2], 0, -axis[0],
        -axis[1], axis[0], 0;
    Eigen::Matrix3f R = cos(angle) * I + (1 - cos(angle)) * axis * axis.transpose() + sin(angle) * N;
    rotation << R(0, 0), R(0, 1), R(0, 2), 0,
        R(1, 0), R(1, 1), R(1, 2), 0,
        R(2, 0), R(2, 1), R(2, 2), 0,
        0, 0, 0, 1;
    return rotation;
}

// 2. 视图变换（移动相机到中心点 + g、t、(g x t)向量移动）
// （这里相机已经在z轴上，只用移到中心点；相机三个向量已正确，不用动）
Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 
	    0, 1, 0, -eye_pos[1], 
		0, 0, 1, -eye_pos[2], 
		0, 0, 0, 1;
		
    view = translate * view;
    
    return view;
}

// 3. 投影变换（透视投影 + 正交投影）
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function
    
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    
    // zNear和zFar都处在z的负轴
    zNear = -zNear;
    zFar = -zFar;
    
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
    
    projection = scale * move * perspective * projection;
    
    return projection;
}
```

### 2. 绘制三角形

1. 每次都通过 mvp矩阵 来将三角形**三个点的坐标进行变换**
2. 将变换后的处在 $[ -1, 1 ]^3$ 的正方体转化为 $[0,width],[0,height]$ 的位置
3. 之后再根据三角形**三个点绘制直线**

```cpp
void rst::rasterizer::draw(rst::pos_buf_id pos_buffer, rst::ind_buf_id ind_buffer, rst::Primitive type)
{
    // 只绘制三角形
    if (type != rst::Primitive::Triangle)
    {
        throw std::runtime_error("Drawing primitives other than triangle is not implemented yet!");
    }

    // buf为包含三个点的vector
    auto& buf = pos_buf[pos_buffer.pos_id];
    // ind为包含一个点的vector，该点的三个值为buf中对应的三个点下标
    auto& ind = ind_buf[ind_buffer.ind_id];

    float f1 = (100 - 0.1) / 2.0;
    float f2 = (100 + 0.1) / 2.0;

    // mvp视图矩阵
    Eigen::Matrix4f mvp = projection * view * model;
    // 只有一次循环
    for (auto& i : ind)
    {
        Triangle t;

        // 1.对三角形三个点都左乘mvp进行视图变换
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };

        // 将点的齐次坐标的最后一个值置1
        for (auto& vec : v) {
            vec /= vec.w();
        }

        // 2.将三角形三个点都移动和缩放到[0,width]、[0,height]、[0,50]之间
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        t.setColor(0, 255.0,  0.0,  0.0);
        t.setColor(1, 0.0  ,255.0,  0.0);
        t.setColor(2, 0.0  ,  0.0,255.0);

        // 3.绘制三角形三条边
        rasterize_wireframe(t);
    }
```

**Bresenham’s algorithm(布兰森汉姆算法)画直线**：根据起始点和终点得到直线，对于每一个 $x_{i+1} = x_{i}+1$ ，$y_{i+1} = y_i / y_i+1$ ，$y_{i+1}$ 的值需要根据 $x_{i+1}$ 带入直线方程中计算得到实际的点的y，看 y 靠近 $y_i$ 还是 $y_{i+1}$，靠近哪一个，就选择哪一个点作为下一个像素点

![[Pasted image 20230710100916.png|300]]

实际代码中需要根据斜率m的情况来判断每次是增加 $x_i$ 还是增加 $y_i$ 

[(83条消息) Bresenham’s algorithm( 布兰森汉姆算法)画直线_七月简语的博客-CSDN博客](https://blog.csdn.net/ShenDW818/article/details/88669209)