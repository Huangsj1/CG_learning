Whitted-Style 的光线追踪到漫反射材质的物体就停止反射光线（直接用Blinn-Phong计算color值），但实际上应该**继续反射**；且Whitted-Style也是直接用了光强 I 来计算color，实际上应该涉及到**辐射度量学**的相关内容，所以就引出了Path-Tracing路径追踪

* 本次实验没有完成微表面模型 Microfacet，等以后如果学了games202再来补了/(ㄒoㄒ)/~~

# 注意要点

1. 需要修改上节课的 `Bounds3::IntersectP()` 函数中的最后的判断：`if (t_enter <= t_exit && t_exit >= 0) return true;`（前面多加了等于号，与object的边相交也要算上，不然图像很黑）
2. 注意 `Intersection` 中的成员 `Vector3f emit` 和 `Material* m`，前者在求光线与物体相交的时候`getIntersection()`和`sample()`随机采样取点的时候会赋值；后者只有求光线与物体相交的时候才会赋值，所以要用**自发光项**的时候用 `emit`，不要用 `m->m_emission`

## 1. 每个像素只采用一次路径追踪

取 `Renderer()` 中的 spp = 1，虽然输出的图像有很多噪点，但是能更**快**渲染出结果（可以先看spp=1时图像是否大致符合再增大spp）

![[Pasted image 20230727171910.png|500]]

```cpp
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    // 先得到与光线相交的物体的交点
    Intersection inter = intersect(ray);
    Vector3f hit_color = this->backgroundColor;
    // 如果没有交点就直接返回背景颜色 
    if (!inter.happened) {
        return hit_color;
    }

    // 如果打到光源就直接返回光照
    if (inter.m->hasEmission()) {
        return inter.m->getEmission();
    }

    Vector3f hitPoint = inter.coords;
    Vector3f N = inter.normal;

    // 一、直接光照 
    Vector3f L_dir;
    Intersection light_inter;
    float light_pdf = 0;
    // 1.对光源进行随机采样
    sampleLight(light_inter, light_pdf);
    // 2.判断路径上是否被其他物体遮挡
    Vector3f reflectionVector = light_inter.coords - inter.coords;
    Vector3f reflectionDirection = reflectionVector.normalized();
    /*Vector3f reflectionOrig = (dotProduct(reflectionDirection, N) > 0) ? hitPoint + N * EPSILON : hitPoint - N * EPSILON;*/
    Vector3f reflectionOrig = hitPoint;
    // 由于这里只有漫反射，所以反射光线的方向一定要与N同向
    // 如果不保证会怎样？？
    if (dotProduct(reflectionDirection, N) > 0) {
        Intersection block_inter = intersect(Ray(reflectionOrig, reflectionDirection));
        // sampleLight和sample得到的Intersection都没有distance值
        if (block_inter.distance > reflectionVector.norm() - EPSILON) {
            // L_dir = L_i * f_r * cosθ * cosθ' / (x - p)^2 / l_pdf
            // 注意这里得到自发光项时不要light_inter.m->m_emission，因为sample的时候没有将m赋给light_inter
            L_dir = light_inter.emit * inter.m->eval(ray.direction, reflectionDirection, N) * dotProduct(reflectionDirection, N)
                * dotProduct(-reflectionDirection, light_inter.normal) / pow(reflectionVector.norm(), 2) / light_pdf;
        }
    }
    
    // 二、间接光照 
    Vector3f L_indir;
    // 1.测试是否通过俄罗斯轮盘(通过概率为P（0.8）)
    if (get_random_float() < RussianRoulette) {
        // 2.随机采样得到反射光线
        Vector3f reflectionDirection = inter.m->sample(ray.direction, N);
        Vector3f reflectionOrig = hitPoint;
        if (dotProduct(reflectionDirection, N) > 0) {
            Intersection block_inter = intersect(Ray(reflectionOrig, reflectionDirection));
            // 3.如果反射光线击中不发光的物体就产生间接光照 
            if (block_inter.happened && !block_inter.m->hasEmission()) {
                L_indir = castRay(Ray(reflectionOrig, reflectionDirection), depth + 1) * inter.m->eval(ray.direction, reflectionDirection, N) * dotProduct(reflectionDirection, N)
                    / inter.m->pdf(ray.direction, reflectionDirection, N) / RussianRoulette;
            }
        }
    }
    return L_dir + L_indir; 
}
```

 ![[Pasted image 20230727172201.png|400]]

* 存在问题：
	1. 出现黑色条纹
	2. 右边箱子的上平面有一个三角形凹进去

### 改进一：修改交点的起始坐标往N的方向一点

```cpp
Vector3f reflectionOrig = hitPoint + N * EPISLON;
```

![[Pasted image 20230727172933.png|400]]

解决了右边物体有点凹进去的问题

### 改进二：增大EPISLON

```cpp
const float EPSILON = 1;
```

![[Pasted image 20230727175006.png|400]]

解决了背景有黑色线条问题

## 2. 每个像素采用16次路径追踪

将ssp设置为16，同时由于太过耗时（估计得半个多小时），根据网上建议将 `get_random_float()` 中的变量设置为 static，时间减少到了四百多秒

![[Pasted image 20230727182057.png|400]]

## 3. 多线程

### 1. std::thread

用 `std::thread` 来将高度分成32份，每份用一个线程来跑

注意：① 线程函数的引用需要在创建 `thread` 的时候对参数显示声明 `std::ref()` ；② 对于引用传参，若没有修改就要加const（这里的scene不加const就会报错）

```cpp
// 一、全局变量定义
int prog = 0;
std::mutex lock;
// 函数声明
void thread_func(Vector3f eye_pos, std::vector<Vector3f>& framebuffer, const Scene& scene, float scale, float imageAspectRatio, int spp, int id, int thread_step);

// 二、主函数
void Renderer::Render(const Scene& scene)
{
    // ...
    // 1.定义与多线程有关的变量
    int thread_num = 32;
    int thread_step = scene.height / thread_num;    // 960 / 32 = 30
    std::vector<std::thread> all_threads;

    // change the spp value to change sample ammount
    int spp = 16;
    std::cout << "SPP: " << spp << "\n";

    // 2.创建并执行多线程
    for (int i = 0; i < thread_num; i++) {
        all_threads.push_back(std::thread(thread_func, eye_pos, std::ref(framebuffer), std::ref(scene), scale, imageAspectRatio, spp, i, thread_step));
    }

    // 3.回收所有线程
    for (int i = 0; i < thread_num; i++) {
        all_threads[i].join();
    }
    UpdateProgress(1.f);
	
	// ... 
}

// 三、线程调用的函数
void thread_func(Vector3f eye_pos, std::vector<Vector3f> & framebuffer, const Scene & scene, float scale, float imageAspectRatio, int spp, int id, int thread_step) 
{
    for (uint32_t j = id * thread_step; j < (id + 1) * thread_step; ++j) {
        for (uint32_t i = 0; i < scene.width; ++i) {
            // generate primary ray direction
            float x = (2 * (i + 0.5) / (float)scene.width - 1) *
                imageAspectRatio * scale;
            float y = (1 - 2 * (j + 0.5) / (float)scene.height) * scale;

            Vector3f dir = normalize(Vector3f(-x, y, 1));

            // 对每个像素点都多次采样，得到多个ωr，每个ωr只对应一个ωi，防止出现指数增长问题
            for (int k = 0; k < spp; k++) {
                framebuffer[j * scene.width + i] += scene.castRay(Ray(eye_pos, dir), 0) / spp;
            }
        }
        // 输出进图条时要加锁，同时用一个全局变量来查看进度
        lock.lock();
        prog++;
        UpdateProgress(prog / (float)scene.height);
        lock.unlock();
    }
}
```

可以看到速度快了两倍多

![[Pasted image 20230728163406.png|400]]

### 2. OpenMP

在对需要多线程执行的for循环前加上 `#pragma omp parallel for`

注意：锁要**初始化**后才能用

```cpp
// OpenMP 多线程
omp_lock_t lock2;
void omp_func(Vector3f eye_pos, std::vector<Vector3f>& framebuffer, const Scene& scene, float scale, float imageAspectRatio, int spp, int id, int thread_step);


// 3.OpenMP多线程
void Renderer::Render(const Scene& scene)
{
	// ...
	
    // 1.定义与多线程有关的变量
    int thread_num = 32;
    int thread_step = scene.height / thread_num;    // 784 / 32 = 24.5
    
    // change the spp value to change sample ammount
    int spp = 16;
    std::cout << "SPP: " << spp << "\n";

    // 初始化锁
    omp_init_lock(&lock2);

    // 2.创建并执行多线程：OpenMP要加下面一句话，可以对for题速
    #pragma omp parallel for
    for (int i = 0; i < thread_num; i++) {
        omp_func(eye_pos, std::ref(framebuffer), std::ref(scene), scale, imageAspectRatio, spp, i, thread_step);
    }
    
    // 回收锁
    omp_destroy_lock(&lock2);

    UpdateProgress(1.f);
	// ...
}


// 线程函数
void omp_func(Vector3f eye_pos, std::vector<Vector3f>& framebuffer, const Scene& scene, float scale, float imageAspectRatio, int spp, int id, int thread_step)
{
    for (uint32_t j = id * thread_step; j < (id + 1) * thread_step; ++j) {
        for (uint32_t i = 0; i < scene.width; ++i) {
            // generate primary ray direction
            float x = (2 * (i + 0.5) / (float)scene.width - 1) *
                imageAspectRatio * scale;
            float y = (1 - 2 * (j + 0.5) / (float)scene.height) * scale;

            Vector3f dir = normalize(Vector3f(-x, y, 1));

            // 对每个像素点都多次采样，得到多个ωr，每个ωr只对应一个ωi，防止出现指数增长问题
            for (int k = 0; k < spp; k++) {
                framebuffer[j * scene.width + i] += scene.castRay(Ray(eye_pos, dir), 0) / spp;
            }
        }
        // 输出进图条时要加锁，同时用一个全局变量来查看进度
        omp_set_lock(&lock2);
        prog++;
        UpdateProgress(prog / (float)scene.height);
        omp_unset_lock(&lock2);
    }
}
```

时间和多线程 `std::thread` 差不多（不像网上说的快很多）

![[Pasted image 20230728171223.png|400]]


## 4. 优化像素点的采集

![[af81e06eb85f21f006802ab824ae794.jpg|300]]

注意这里的横坐标 $x = \frac{dw}{2} + i\%step\cdot dw$ （其中 i 为第 i 个点），而纵坐标 $y = \frac{dw}{2} + i / step\cdot dw$ 

```cpp
void Renderer::Render(const Scene& scene)
{
	// ...
	for (uint32_t j = 0 ; j < scene.height; ++j) {
		for (uint32_t i = 0; i < scene.width; ++i) {
			// 对每个像素点都多次采样，得到多个ωr，每个ωr只对应一个ωi，防止出现指数增长问题
			for (int k = 0; k < spp; k++) {
				float x = (2 * (i + dw / 2 + (k % step) * dw) / (float)scene.width - 1) *
					imageAspectRatio * scale;
				float y = (1 - 2 * (j + dw / 2 + (k / step) * dw) / (float)scene.height) * scale;
				
				Vector3f dir = normalize(Vector3f(-x, y, 1));
				framebuffer[j * scene.width + i] += scene.castRay(Ray(eye_pos, dir), 0) / spp;
			}
		}
		UpdateProgress(j / (float)scene.height);
	}
	// ...
}
```

结果其实和上面的图片差不多

![[Pasted image 20230728180224.png|400]]