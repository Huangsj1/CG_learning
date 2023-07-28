# 一、包围盒与光线求交

通过[[5.Ray Tracing光线追踪#3. 光线物体表面相交的加速（AABB包围盒方法）|AABB包围盒]]方法，根据光线与包围盒三个轴方向交点得到 $t_{enter}$ 和 $t_{exit}$ ，通过判断 $t_{enter} < t_{exit}$ && $t_{exit} \geq 0$ 来判断与光线相交

```cpp
// 判断光线是否与包围盒相交
inline bool Bounds3::IntersectP(const Ray& ray, const Vector3f& invDir, const std::array<int, 3>& dirIsNeg) const
{
    float t_enter = -kInfinity;
    float t_exit = kInfinity;
    // 遍历三个轴求各自的交点
    for (int i = 0; i < 3; i++) {
        // 光线如果是正向，pMin为靠近的点，pMax为远处的点
        if (dirIsNeg[i]) {
            t_enter = std::max(t_enter, float((pMin[i] - ray.origin[i]) * invDir[i]));
            t_exit = std::min(t_exit, float((pMax[i] - ray.origin[i]) * invDir[i]));
        } // 光线是反向则相反
        else {
            t_enter = std::max(t_enter, float((pMax[i] - ray.origin[i]) * invDir[i]));
            t_exit = std::min(t_exit, float((pMin[i] - ray.origin[i]) * invDir[i]));
        }
    }
    if (t_enter < t_exit && t_exit >= 0) return true;
    return false;
}

```

# 二、将包围盒构建树形结构来遍历

![[6ad34577a0c0322d2f8b3315f416be2.jpg|400]]

## 1. BVH

>每次划分都是沿着包围盒**最长的轴**按**objects的数量平均**划分左右子树

1. `recursiveBuild()` 递归构建BVH树
	1. 当只有一个object时：当前节点（叶子节点）BVHBuildNode 就包含 object ，左右子树为 nullptr
	2. 当只有两个objects时：递归左右子树各一个object
	3. 多于两个objects时：将包含所有物体的包围盒的**最长的轴**作为分割轴，经过排序后，左右子树**各一半 objects** 递归
2. `getIntersection()` 求光线的交点：如果与当前包围盒有交点
	1. 如果当前包围盒是叶子节点（有object），直接返回与物体的交点 `object->getIntersection(ray)`
	2. 否则递归得到左右包围盒的交点，选择距离 `distance` 近的 Intersection返回

```cpp
Intersection BVHAccel::getIntersection(BVHBuildNode* node, const Ray& ray) const
{
    // TODO Traverse the BVH to find intersection
    std::array<int, 3> dirIsNeg = { ray.direction[0] > 0, ray.direction[1] > 0, ray.direction[2] > 0 };

    // 如果与包围盒没有交点就直接返回带false的Intersection
    if (!node->bounds.IntersectP(ray, ray.direction_inv, dirIsNeg))
        return Intersection();
    // 如果是叶子节点（有object 且 left和right均为nullptr）
    if (node->object != nullptr)
        return node->object->getIntersection(ray);

    Intersection l = getIntersection(node->left, ray);
    Intersection r = getIntersection(node->right, ray);
    return l.distance < r.distance ? l : r;
}
```

![[Pasted image 20230726121705.png]]

## 2. SAH

[SAH加速参考链接](https://zhuanlan.zhihu.com/p/50720158)

> 构建启发函数——表面积启发函数，来使得划分更加合理（左右包围盒重叠面积小），Render时更快

```cpp
// 每次选取包含所有中心点的包围盒中最长的轴作为划分轴，通过快排将物体分成两堆，递归构建左右子树
// 叶子节点（BVHBuildNode）才有object
BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*> objects)
{
    BVHBuildNode* node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    // bounds是最大的包围盒
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    // 1.如果包围盒中只有一个物体，其object就是该物体
    if (objects.size() == 1) {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }
    // 2.如果包围盒有两个物体，
    else if (objects.size() == 2) {
        node->left = recursiveBuild(std::vector{objects[0]});
        node->right = recursiveBuild(std::vector{objects[1]});

        // 这里可以直接等于bounds
        /*node->bounds = Union(node->left->bounds, node->right->bounds);*/
        node->bounds = bounds;
        return node;
    }
    // 3.如果包围盒有多个物体，递归左子树和右子树
    else {
        // 包含所有object的包围盒中心点的包围盒
        Bounds3 centroidBounds;
        for (int i = 0; i < objects.size(); ++i)
            centroidBounds =
                Union(centroidBounds, objects[i]->getBounds().Centroid());
        // 看哪一个维度最大就选哪个维度来划分（通过快排取中心点来划分左右子树）
        int dim = centroidBounds.maxExtent();
        switch (dim) {
        case 0:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().x <
                       f2->getBounds().Centroid().x;
            });
            break;
        case 1:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().y <
                       f2->getBounds().Centroid().y;
            });
            break;
        case 2:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().z <
                       f2->getBounds().Centroid().z;
            });
            break;
        }

        auto beginning = objects.begin();
        auto middling = objects.begin() + (objects.size() / 2);
        auto ending = objects.end();

        // 如果用了SAH加速
        if (splitMethod == SplitMethod::SAH) {
            // 桶的数量(过多就为10，同时保证最少为2)
            int bucketNum = objects.size() > 10 ? 10 : objects.size() / 2 + 1;
            /*int bucketNum = objects.size();*/
            int bestId = 0;
            float minCost = kInfinity;
            float SC = bounds.SurfaceArea();
            float tTrav = 0.125;
            for (int i = 1; i < bucketNum; i++) {
                // 1.根据桶的数量划分为左右两部分
                middling = objects.begin() + objects.size() * i / bucketNum;
                auto leftshapes = std::vector<Object*>(beginning, middling);
                auto rightshapes = std::vector<Object*>(middling, ending);
                assert(objects.size() == (leftshapes.size() + rightshapes.size()));
                
                // 2.分别计算两部分包围盒的面积
                Bounds3 leftBounds, rightBounds;
                for (int k = 0; k < leftshapes.size(); k++) {
                    leftBounds = Union(leftBounds, leftshapes[k]->getBounds());
                }
                for (int k = 0; k < rightshapes.size(); k++) {
                    rightBounds = Union(rightBounds, rightshapes[k]->getBounds());
                }
                float SA = leftBounds.SurfaceArea();
                float SB = rightBounds.SurfaceArea();

                // 3.更新最小cost
                float nowCost = SA / SC * leftshapes.size() + SB / SC * rightshapes.size() + tTrav;
                if (nowCost < minCost) {
                    minCost = nowCost;
                    bestId = i;
                }
            }
            middling = objects.begin() + objects.size() * bestId / bucketNum;
        }

        auto leftshapes = std::vector<Object*>(beginning, middling);
        auto rightshapes = std::vector<Object*>(middling, ending);

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));

        node->left = recursiveBuild(leftshapes);
        node->right = recursiveBuild(rightshapes);

        node->bounds = Union(node->left->bounds, node->right->bounds);
    }

    return node;
}
```

### 1. 所有物体都作为一个桶来遍历

如果对所有物体都单独划分（桶的数量为物体个数），RecursiveBuild构建树时间长，但是Render时间短了一点

```cpp
int bucketNum = objects.size();
```

![[Pasted image 20230726122815.png]]

### 2. 限制桶的最大数量（10）

桶的数量最多为10个时，构建树时间相对BVH多了一点，Render的时间和BVH差不多

```cpp
// 桶的数量(过多就为10，同时保证最少为2)
int bucketNum = objects.size() > 10 ? 10 : objects.size() / 2 + 1;
```

![[Pasted image 20230726123052.png]]