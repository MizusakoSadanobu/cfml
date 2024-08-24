# Chapter 0: 基礎知識
## 教師あり学習
特徴量 $x$ から目的変数 $y$ を求めるための関数 $y=f(x)$ を推定する問題。
- 理想的に解きたい問題

```math
f^{*}=\text{argmin}_{f \in F}L(f)
```

```math
L(f) := \text{E}_{p(x,y)}[(y-f(x))^2]
```

- 実際に解く問題

```math
\hat{f}=\text{argmin}_{f \in F}\hat{L}(f;D)
```

```math
\hat{L}_{AVG}(f;D):=\frac{1}{n} \sum_{i=1}^n l(y_i, f(x_i))
```

典型的には、$\hat{L}_{AVG}(f;D)$ がよく用いられるが、これが本当に適切なのかを疑う必要あり。

**平均介入効果（ATE: Average Treatment Effect）：**

```math
\tau := \text{E}_{p(y(1),y(0))}[y(1)-y(0)] = \text{E}_{p(y(1))}[y(1)] - \text{E}_{p(y(0))}[y(0)]
```

これが機能するための前提：
- 一致性：介入を受けた際には $y(1)$ を観測し、介入を受けなかった際には $y(0)$ を観測する。
- 交換性：介入有無は特徴量で条件づけた時に独立になっている。→未観測交絡が存在するとこの仮定が崩れる
- 正値性：介入を受ける確率、受けない確率はどちらもゼロではない。

**観測データの経験平均に基づくAVG推定量：**

```math
\hat{\tau}_{AVG}(D):=\frac{1}{n}\sum_{i=1}^n w_i y_i - \frac{1}{n}\sum_{i=1}^n (1-w_i) y_i
```

上記の $\hat{\tau}_{AVG}(D)$ と $\tau $ は一致するのか？

```math
\hat{\tau}_{AVG}(D)=\text{E}_{p(x,y(1))}[e(x)y(1)]-\text{E}_{p(x,y(0))}[(1-e(x))y(0)]
```

```math
\neq \tau
```

ただし、$e(x):=\text{E}_{p(w|x)}[w]$、介入を受ける確率を表す。

$e(x)$によって$\hat{\tau}_{AVG}(D)$はバイアスを受けてしまう。<br>
→では、どのように補正するのか？<br>
→**IPS** (Inverse Propensity Score: IPS)による補正。

**IPS推定量：**

```math
\hat{\tau}_{IPS}(D) := \frac{1}{n} \sum_{i=1}^{n} \frac{w_i}{e(x_{i})}y_{i} - \frac{1}{n} \sum_{i=1}^{n} \frac{(1-w_i)}{(1-e(x_{i}))}y_{i}
```