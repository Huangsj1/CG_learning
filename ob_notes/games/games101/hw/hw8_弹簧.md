本次作业主要用的是[[10.Animation-Simulation#1. Mass Spring System 质点弹簧系统|质点弹簧系统]]

* 注意：对于质点 `mass` 受到的力需要 +=，而且需要在模拟完力和运动后清除力：`m->forces = Vector2D(0, 0);`，因为每次运动后受力情况都要重新分析

# 一、绳子的创建

```cpp
Rope::Rope(Vector2D start, Vector2D end, int num_nodes, float node_mass, float k, vector<int> pinned_nodes)
{
	// 每一段的长度为step
	Vector2D step = (end - start) / (num_nodes - 1);
	for (int i = 0; i < num_nodes; i++)
	{
		// 1.创建节点
		masses.push_back(new Mass(start + i * step, node_mass, true));
		if (i > 0)
		{
			// 2.创建弹簧，且只有第一个弹簧是固定的
			springs.push_back(new Spring(masses[i - 1], masses[i], k));
			masses[i]->pinned = false;
		}
	}
}
```

# 二、力和运动

## 一、欧拉法

```cpp
void Rope::simulateEuler(float delta_t, Vector2D gravity)
{
	float kd = 0.05;
	float kad = 0.005;
	for (auto &s : springs)
	{
		// TODO (Part 2): Use Hooke's law to calculate the force on a node
		// 1.先用胡克定律求每个节点受到的力，累加起来
		// 其中dir1to2和force都是从1指向2，所以下面m1的力直接加，m2的力直接减
		Vector2D dir1to2 = s->m2->position - s->m1->position;
		float length = dir1to2.norm();
		Vector2D force = s->k * (dir1to2 / length) * (length - s->rest_length);
		s->m1->forces += force;
		s->m2->forces -= force;

		// 2.添加质点之间的抑制力damping
		Vector2D relative_velocity = s->m2->velocity - s->m1->velocity;
		Vector2D damping_force = kd * dot((dir1to2 / length), relative_velocity) * (dir1to2 / length);
		s->m1->forces += damping_force;
		s->m2->forces -= damping_force;

		// 3.添加空气阻力
		s->m1->forces -= kad * s->m1->velocity;
		s->m2->forces -= kad * s->m2->velocity;
	}

	for (auto &m : masses)
	{
		if (!m->pinned)
		{
			// TODO (Part 2): Add the force due to gravity, then compute the new velocity and position
			m->forces += gravity;
			Vector2D a = m->forces / m->mass;
			// 1.显示欧拉法
			// m->position = m->position + m->velocity * delta_t;
			// m->velocity = m->velocity + a * delta_t;

			// 2.隐式欧拉法
			m->velocity = m->velocity + a * delta_t;
			m->position = m->position + m->velocity * delta_t;

			// TODO (Part 2): Add global damping
		}

		// Reset all forces on each mass
		m->forces = Vector2D(0, 0);
	}
}
```

### 1. 力的模拟

#### 1. 弹簧的胡克定律

胡克定律得到弹力 -> 一开始效果较好，后面绳子很乱

```cpp
void Rope::simulateEuler(float delta_t, Vector2D gravity)
{
	// ...
	for (auto &s : springs)
	{
		// 1.先用胡克定律求每个节点受到的力，累加起来
		// 其中dir1to2和force都是从1指向2，所以下面m1的力直接加，m2的力直接减
		Vector2D dir1to2 = s->m2->position - s->m1->position;
		float length = dir1to2.norm();
		Vector2D force = s->k * (dir1to2 / length) * (length - s->rest_length);
		s->m1->forces += force;
		s->m2->forces -= force;
		// ...
	}
	// ...
}
```

#### 2. 物体重力

```cpp
void Rope::simulateEuler(float delta_t, Vector2D gravity)
{
	// ...
	for (auto &m : masses)
	{
		// ...
		if (!m->pinned)
		{
			m->forces += gravity;
			// ...
		}
		//...
	}
	// ...
}
```

#### 3. 弹簧的质点间抑制力damping

添加质点之间的抑制力damping -> 绳子摆动较正常，但是会一直摆动（总体能量没有损失）

注意相对速度那里是B相对A的速度，且[[10.Animation-Simulation#1. 两个质点的相互作用力|相互抑制力公式的红色方框]]中是相对速度在运动方向上的分量，需要点乘`dot()`

```cpp
void Rope::simulateEuler(float delta_t, Vector2D gravity)
{
	// ...
	for (auto &s : springs)
	{
		// ...
		// 2.添加质点之间的抑制力damping
		Vector2D relative_velocity = s->m2->velocity - s->m1->velocity;
		Vector2D damping_force = kd * dot((dir1to2 / length), relative_velocity) * (dir1to2 / length);
		s->m1->forces += damping_force;
		s->m2->forces -= damping_force;
		// ...
	}
	// ...
}
```

#### 4. 空气阻力 

添加空气阻力air_damping -> 绳子慢慢停止

```cpp
void Rope::simulateEuler(float delta_t, Vector2D gravity)
{
	// ...
	for (auto &s : springs)
	{
		// ...
		// 3.添加空气阻力
		s->m1->forces -= kad * s->m1->velocity;
		s->m2->forces -= kad * s->m2->velocity;
		// ...
	}
	// ...
}
```

### 2. 运动公式

#### 1. 显示欧拉法

绳子会跑飞，不好

```cpp
void Rope::simulateEuler(float delta_t, Vector2D gravity)
{
	for (auto &m : masses)
	{
		if (!m->pinned)
		{
			m->forces += gravity;
			Vector2D a = m->forces / m->mass;
			// 1.显示欧拉法
			m->position = m->position + m->velocity * delta_t;
			m->velocity = m->velocity + a * delta_t;

			// 2.隐式欧拉法
			// m->velocity = m->velocity + a * delta_t;
			// m->position = m->position + m->velocity * delta_t;
		}
		// Reset all forces on each mass
		m->forces = Vector2D(0, 0);
	}
}
```

#### 2. 隐式欧拉法

绳子可以正常运动

```cpp
void Rope::simulateEuler(float delta_t, Vector2D gravity)
{
	for (auto &m : masses)
	{
		if (!m->pinned)
		{
			m->forces += gravity;
			Vector2D a = m->forces / m->mass;
			// 1.显示欧拉法
			// m->position = m->position + m->velocity * delta_t;
			// m->velocity = m->velocity + a * delta_t;

			// 2.隐式欧拉法
			m->velocity = m->velocity + a * delta_t;
			m->position = m->position + m->velocity * delta_t;
		}
		// Reset all forces on each mass
		m->forces = Vector2D(0, 0);
	}
}
```

## 二、Verlet 模拟运动

Verlet相对于上面的欧拉法，受到的力也有[[#1. 弹簧的胡克定律|胡克定律]]和[[#2. 物体重力|物体重力]]，运动公式通过 `x(t + 1) = x(t) + [x(t) - x(t-1)] + a(t)*dt*dt` 来更新，同时为了模拟上面欧拉法的阻力（弹簧间抑制力和空气阻力）更新了运动公式：`x(t + 1) = x(t) + (1 - damping_factor) * [x(t) - x(t-1)] + a(t)*dt*dt`

不要忘了在物体运动后重置力 `m->forces = Vector2D(0, 0);`

```cpp
void Rope::simulateVerlet(float delta_t, Vector2D gravity)
{
	float damping_factor = 0.0005;
	for (auto &s : springs)
	{
		// 胡克定律
		Vector2D dir1to2 = s->m2->position - s->m1->position;
		float length = dir1to2.norm();
		Vector2D force = s->k * (dir1to2 / length) * (length - s->rest_length);
		s->m1->forces += force;
		s->m2->forces -= force;
	}

	for (auto &m : masses)
	{
		if (!m->pinned)
		{
			Vector2D temp_position = m->position;
			// 物体重力
			m->forces += gravity;
			Vector2D a = m->forces / m->mass;
			
			// 运用Verlet非物理方法模拟x,同时加入了阻尼模拟摩擦力
			m->position = m->position + (1 - damping_factor) * (m->position - m->last_position) + a * delta_t * delta_t;
			m->last_position = temp_position;
		}
		// Reset all forces on each mass
		m->forces = Vector2D(0, 0);
	}
}
```


最后结果中绿色为Verlet方法的，蓝色为欧拉方法（受力分析的）

![[Pasted image 20230728231548.png]]