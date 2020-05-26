---
title: 벡터공간
author: Monch
category: linear_algebra
layout: post
---

[선형대수 LIST로 가기](https://songminkee.github.io/linear_algebra/2030/05/03/list.html)

<h2>벡터공간(VectorSpace)</h2>

체(field) $$F$$에 대한 가군 $$(V,+,\cdot)$$ 을 벡터공간, $$V$$의 원소를 벡터라 한다.  
이때 $$+$$는 벡터의 덧셈이고, $$\cdot$$는 벡터의 Scalar배이다. 



<h3>(1) 벡터의 공리</h3>  

- $$(V,+)$$는 아벨군이다 $$(u,v,w \in V)$$.
  - $$(u+v)+w=u+(v+w)$$.
  - $$u+v=v+u$$.
  - $$u + \vec{0} = u$$ 인 $$\vec{0}$$가 $$V$$에 존재한다.
  - $$u + ( - u ) = \vec{0}$$ 인 $$-u$$가 $$V$$에 존재한다.
- $$(V,+,\cdot)$$는 $$F$$의 가군이다. $$(k , m \in F)$$.
  - $$k \cdot (m \cdot u) = (km) \cdot u$$.
  - $$F$$의 곱셈 항등원 $$1$$에 대해 $$1 \cdot u = u$$.
  - $$(k_m) \cdot (u+v) = k \cdot u + m \cdot u + k \cdot v + m \cdot v$$.



<h3>(2) 선형 생성(Linear Span)</h3>

<h4>1) 부분벡터공간</h4>  
벡터공간 $$V$$상에서 정의된 덧셈과 스칼라배에 대하여 그 자체로서 벡터공간이 되는 $$V$$의 부분집합 $$W$$를 $$V$$의 부분벡터공간 또는 부분공간이라 한다.

<h4>2) 선형생성</h4>  
벡터공간 $$V$$의 공집합이 아닌 부분집합 $$S={v_{1},v_{2}, \ ... \ ,v_{n}}$$ 내의 벡터들의 가능한 모든 선형결합으로 이루어진, $$V$$의 부분벡터공간을 $$S$$의 (선형) 생성 $$span(S)$$이라 한다.
즉,
$$
span(S) = \begin{Bmatrix} \sum_{i=0}^{m} k_{i}v_{i} | \ k_{i} \in F , v_{i} \in S \end{Bmatrix}
$$
이때 $$S$$가 $$span(S)$$을 (선형)생성한다라고 한다.

ex)
$$S=\begin{Bmatrix}(1,0),(0,1)\end{Bmatrix}$$.
$$F=\mathbb{R}$$.
$$\Rightarrow span(S) = \begin{Bmatrix}k(1,0)+m(0,1) \ | \ k,m \in F \end{Bmatrix}$$.
$$= \begin{Bmatrix}k(1,0)+m(0,1) \ | \ k,m \in F \end{Bmatrix}$$.
$$= \mathbb{R^{2}}$$.



<h3>(3) 선형독립(Linear_independent)</h3>

다음을 만족 할 때, 벡터 $$a_{1}, ... ,a_{n}$$은 선형독립(일차독립 혹은 독립)이라고 한다.  
수 $$u_{1}, ... , u_{n}$$에 대해  
$$u_{1}a_{1}+...+u_{n}a_{n}=\mathbf{0}$$라면  
'$$u_{1}=...=u_{n}=0$$'  
이를 반대로 생각했을 때 벡터(혹은 수의 집합) $$x_1 \ne x_2$$일 때, $$x_{1}-x_{2} \ne 0$$이 된다. 즉 결과 $$y$$에 대한 유일한 표현이 된다.  
조건을 만족 못했을때 $$a{1}, ... ,a{n}$$을 선형종속(일차종속 혹은 종속)이라고 한다.  
선형종속이 포함된 사상 $$A$$는 정방행렬이어도 차원을 감소시키는 사상이 된다.

ex)  

- $$S_{1}=\begin{Bmatrix}(1,0),(0,1),(1,1)\end{Bmatrix}$$.
  $$k_{1}(1,0)+k_{2}(0,1)+k_{3}(1,1)=\vec{0}$$.
  $$\Rightarrow \left( {k_{1}=k_{2}=k_{3}=0 \\ k_{1}=k_{2}=1 , k_{3}=-1} \right.$$.
  $$S_{1}$$는 선형종속 집합



- $$S_{1}=\begin{Bmatrix}(1,0),(0,1)\end{Bmatrix}$$.
  $$k_{1}(1,0)+k_{2}(0,1)=\vec{0}$$.
  $$\Rightarrow k_{1}=k_{2}=0 $$._
  $$S_{2}는 선형독립 집합

<br>

<h3>(4) 여러 벡터공간</h3>

<h4>1) 노름공간(Norm space)</h4>
노름이 부여된 $$K-$$벡터공간 $$(V,\begin{Vmatrix} \cdot \end{Vmatrix})$$  
노름이란 $$\forall u, v \in V, \forall k \in K$$에 대해 아래 세 조건을 만족시키는 함수  
$$\begin{Vmatrix} \cdot \end{Vmatrix}: V \rightarrow [0, \infin)$$이다. $$(K \in \begin{Bmatrix}R,C\end{Bmatrix} )$$, 여기서 $$R$$은 실수집합 $$C$$는 복소수집합.

- $$\begin{Vmatrix} kv \end{Vmatrix} = \begin{vmatrix} k \end{vmatrix} \begin{Vmatrix} v \end{Vmatrix}$$.
- $$\begin{Vmatrix} u+v \end{Vmatrix} \leqq \begin{Vmatrix} u \end{Vmatrix} + \begin{Vmatrix} v \end{Vmatrix}$$.
- $$\begin{Vmatrix} v \end{Vmatrix} = 0 \Leftrightarrow v= \vec{0}$$.



<h4>2) 내적공간</h4>
내적이 부여된 $$K-$$벡터공간 $$(V,\left\langle \cdot,\cdot \right\rangle)$$.  
내적이란 $$\forall u, v, w \in V, \forall k \in K$$에 대해 아래 네 조건을 만족시키는 함수 $$\left\langle \cdot,\cdot \right\rangle : V \times V \rightarrow K$$이다. $$ (K\in \left\{ {R,C} \right\})$$.

- $$\left\langle u+v,w \right\rangle = \left\langle u,w \right\rangle + \left\langle u,w \right\rangle$$.
- $$\left\langle ku,v \right\rangle = k\left\langle v,u \right\rangle$$.
- $$\left\langle u,v \right\rangle = \left\langle \overline{v,u} \right\rangle$$.
- $$v \neq \vec{0} \Rightarrow \left\langle v,v \right\rangle > 0$$.



<h4>3) 유클리드 공간</h4>

음이 아닌 정수 $$n$$에 대하여 $$n$$차원 유클리드공간 $$\mathbb{R^{n}}$$은 실수집합 $$\mathbb{R}$$의 $$n$$번 곱집합이며, 이를 $$n$$차원 실수 벡터공간으로써 정의하기도 한다.
내적 $$\left\langle u,v \right\rangle=\sum_{i=1}^{n}u_{i}v_{i}=u \cdot v$$을 정의하면 점곱, 스칼라곱 이라고도한다.



<h4>4) 기저와 차원</h4>

1 - 기저  
벡터공간 $$V$$의 부분집합 $$B$$가 선형독립이고 $$V$$를 생성할 때, $$B$$를 V의 기저라 한다.

ex)

- $$B_{1} = \left\{ (1,0) ,(0,1)\right\}$$.
  $$\Rightarrow span(B_{1}) = \mathbb{R^2}$$.  
  따라서 $$B_{1}$$은 $$V$$의 기저이다.
- $$B_{2} = \left\{ (1,0) ,(1,1) \right\}$$.
  $$(a,b) = k(1,0)+m(1,1) = (k+m,m)$$._
  $$m=b, k=a-b$$.  
  따라서 $$B_{2}$$는 $$V$$의 기저이다.
- $$S= \left\{ (1,0), (0,1) , (1,1)\right\}$$.
  $$span(S) = \mathbb{R^2}$$.  
  성립하지만 선형종속이다. 따라서 $$S$$는 $$V$$의 기저가 아니다.



2 - 차원  
$$B$$가 벡터공간 $$V$$의 기저일 때 $$B$$의 원소의 개수를 $$V$$의 차원 $$dim(V)$$라 한다.



ex)

- 기저의 예제에서 $$dim(V)=n(B_{1})=n(B_{2})=2$$.

<br>

3 - 정규기저  
다음 조건을 만족하는 노름공간 $$V$$의 기저 $$B$$를 정규기저라 한다.
$$
\forall b \in B, \begin{Vmatrix}b\end{Vmatrix}=1
$$

<br>

4 - 직교기저  
다음 조건을 만족하는 내적공간 $$V$$의 기저 $$B$$를 직교기저라 한다.
$$
\forall b_{1},b_{2} \in B, \left\langle b_{1},b_{2} \right\rangle =0
$$

<br>

5 - 정규직교기저  
정규기저이자 직교기저인 내적공간의 기저를 정규직교기저라 한다.  
특히 $$\mathbb{R^{n}}$$의 정규직교기저 $$\left\{ (1,0,...,0),(0,1,...,0),(0,0,...,1) \right\}를 표준기저라 한다.

ex)  
$$\mathbb{R^2}$$에 대해서

- $$B_{1}= \left\{ (2,0),(1,1) \right\}$$  정규 X, 직교 X
- $$B_{2}= \left\{ (1,0),(\frac{1}{\sqrt{2}},\frac{1}{\sqrt{2}}) \right\}$$  정규 O, 직교 X
- $$B_{3}= \left\{ (1,1),(1,-1) \right\}$$  정규 X, 직교 O
- $$B_{4}= \left\{ (1,0),(0,1) \right\}$$  정규 O, 직교 O



