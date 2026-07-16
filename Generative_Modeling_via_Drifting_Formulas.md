# Generative Modeling via Drifting：公式索引

以下公式对应 Generative_Modeling_via_Drifting.md 正文中的“式 1”至“式 6”。

## 式 1：生成器诱导的分布

$$
q_\theta=(f_\theta)_\#p_\epsilon.
$$

## 式 2：推理时的分布运输

$$
\frac{\mathrm d x_t}{\mathrm d t}=v_t(x_t),\qquad
\partial_t q_t+\nabla\cdot(q_t v_t)=0.
$$

## 式 3：漂移更新

$$
x_{i+1}=x_i+V_{p,q_i}(x_i),\qquad
x_i=f_{\theta_i}(\epsilon).
$$

## 式 4：零漂移平衡

$$
p=q\quad\Longrightarrow\quad V_{p,q}(x)=0.
$$

反对称性：

$$
V_{p,q}(x)=-V_{q,p}(x).
$$

## 式 5：核化漂移场

$$
\begin{aligned}
V^+_p(x)&=\frac{\mathbb E_p[k(x,y^+)(y^+-x)]}{\mathbb E_p[k(x,y^+)]},\\
V^-_q(x)&=\frac{\mathbb E_q[k(x,y^-)(y^--x)]}{\mathbb E_q[k(x,y^-)]},\\
V_{p,q}(x)&=V^+_p(x)-V^-_q(x).
\end{aligned}
$$

其中，$y^+\sim p$ 为数据样本，$y^-\sim q$ 为当前生成样本。

## 式 6：Drifting 训练目标

$$
\mathcal L_{\mathrm{drift}}=
\mathbb E_\epsilon\left[
\left\|f_\theta(\epsilon)-
\operatorname{sg}\left(
f_\theta(\epsilon)+V_{p,q_\theta}(f_\theta(\epsilon))
\right)\right\|_2^2
\right].
$$

其中，$\operatorname{sg}$ 表示 stop-gradient。
