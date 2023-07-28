参考[[4.Geometry#2. 贝塞尔曲线的代数表示|贝塞尔曲线的绘制公式]]即可得到

## 1. 基础部分

**直接**通过递归绘制的曲线

```cpp
// 递增t来绘制贝塞尔曲线
void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    int near_num = 1;
    cv::Vec3b color = { 0, 255, 0 };

    double dt = 0.001;
    for (double t = 0; t <= 1.0; t += dt) {
        auto point = recursive_bezier(control_points, t);
        window.at<cv::Vec3b>(point.y, point.x) = color;
    }
}

// 递归求得t下的点的位置
cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, double t) 
{
    // 如果只剩最后一条线段（两个点）就直接返回位于t位置的点
    if (control_points.size() == 2) {
        return (1 - t) * control_points[0] + t * control_points[1];
    }

    // 两两线段合成一个新的线段，递归
    std::vector<cv::Point2f> sub_points;
    for (int i = 0; i < control_points.size() - 1; ++i) {
        auto point = (1 - t) * control_points[i] + t * control_points[i + 1];
        sub_points.push_back(point);
    }
    return recursive_bezier(sub_points, t);
}
```

![[my_bezier_curve.png|300]] ![[Pasted image 20230720203554.png|300]] 
## 2. 反走样 / 抗锯齿

* 我这里的距离用的都是**曼哈顿距离**（计算量小，效果差不多）

### 1. 以点所在的像素为中心，绘制周围正方形八个点

对曲线中每个像素的**所有周围**像素都按一定**比例衰减**着色 $color = \frac{color}{1 + |dx| + |dy|}$

![[my_bezier_curve 1.png|300]] ![[Pasted image 20230720203643.png|300]]

发现颜色较为暗淡，是因为后面衰减后较暗的颜色**覆盖**了前面较亮的颜色，对屏幕显示的颜色取最大值 `max` （这里也可以取平均值，但是需要额外计数就比较耗时）

![[my_bezier_curve 2.png|300]] ![[Pasted image 20230720203712.png|300]]

```cpp
void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    int near_num = 1;
    int color = 255;

    double dt = 0.001;
    for (double t = 0; t <= 1.0; t += dt) {
        auto point = recursive_bezier(control_points, t);
        int px = point.x;
        int py = point.y;
        // 将相邻near_num距离的点颜色渐变，反走样
        for (int x = px - near_num; x <= px + near_num; x++) {
            for (int y = py - near_num; y <= py + near_num; y++) {
                // 如果在屏幕内就绘制颜色
                if (x >= 0 && x < 700 && y >= 0 && y < 700) {
                    window.at<cv::Vec3b>(y, x)[1] = std::max(int(window.at<cv::Vec3b>(y, x)[1]), color / (1 + std::abs(x - px) + std::abs(y - py)));
                }
            }
        }
    }
}
```

### 2. 根据求得的点的位置与像素中心距离来选取四个点

![[Pasted image 20230720203947.png]]

可以看出这种方法和上面直接求周围八个像素结果差不多，但是需要**显示的像素少了**，且其可以根据实际位置偏移来**更准确选择周围像素**着色

![[my_bezier_curve 3.png|300]] ![[Pasted image 20230720203811.png|300]]

```cpp
// 递增t来绘制贝塞尔曲线
void bezier(const std::vector<cv::Point2f>& control_points, cv::Mat& window)
{
    int color = 255;
    // 像素中心点坐标
    float x0, x1, y0, y1;

    double dt = 0.001;
    for (double t = 0; t <= 1.0; t += dt) {
        auto point = recursive_bezier(control_points, t);
        float px = int(point.x) + 0.5;
        float py = int(point.y) + 0.5;
        // 1.先确定四个像素点位置
        // 偏左边
        if (point.x - px < 0) {
            x0 = px > 0 ? px - 1 : px;
            x1 = px;
        }
        // 偏右边
        else {
            x0 = px;
            x1 = px < 700 - 1 ? px + 1 : px;
        }
        // 偏下面
        if (point.y - py < 0) {
            y0 = py > 0 ? py - 1 : py;
            y1 = py;
        }
        // 偏上面
        else {
            y0 = py;
            y1 = py < 700 - 1 ? py + 1 : py;
        }
        // 2.根据距离衰减填充颜色(曼哈顿距离)
        float d_min = std::abs(point.x - px) + std::abs(point.y - py);
        window.at<cv::Vec3b>(y0, x0)[1] = std::max(float(window.at<cv::Vec3b>(y0, x0)[1]), color * d_min / (std::abs(point.x - x0) + std::abs(point.y - y0)));
        window.at<cv::Vec3b>(y0, x1)[1] = std::max(float(window.at<cv::Vec3b>(y0, x1)[1]), color * d_min / (std::abs(point.x - x1) + std::abs(point.y - y0)));
        window.at<cv::Vec3b>(y1, x0)[1] = std::max(float(window.at<cv::Vec3b>(y1, x0)[1]), color * d_min / (std::abs(point.x - x0) + std::abs(point.y - y1)));
        window.at<cv::Vec3b>(y1, x1)[1] = std::max(float(window.at<cv::Vec3b>(y1, x1)[1]), color * d_min / (std::abs(point.x - x1) + std::abs(point.y - y1)));
    }
}
```