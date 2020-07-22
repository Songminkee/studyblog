---
title: 결합분포, 주변분포, 조건부분포
author: Monch
category: Statistics
layout: post
---

<h2>결합 분포</h2>

확률 변수 X,Y에 대해 $$W \equiv (X,Y)$$이 정의 될 때, W를 X,Y의 결합분포라고 부른다.



<h2>이산값(확률) VS 실수값(확률밀도)</h2>

|                    |                         이산값(확률)                         |                       실수값(확률밀도)                       |
| ------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 주변분포           |               $$P(X=a) = \sum_{y} P(X=a,Y=y)$$               |     $$f_{X}(a) = \int^{\infty}_{-\infty}f_{X,Y}(a,y)dy$$     |
| 조건부분포         | $$P(Y=b \mid X=a) \equiv \frac{P(X=a,Y=b)}{P(X=a)} \\ P(X=a,Y=b) = P(Y=b \mid X=a) P(X=a)$$ | $$f_{Y \mid X}(b \mid a) \equiv \frac{f_{X,Y}(a,b)}{f_{X}(a)} \\ f_{X,Y}(a,b) = f_{Y \mid X} (b \mid a)f_{X}(a)$$ |
| 베이즈 공식        | $$P(X=a \mid Y=b) = \frac{P(Y=b \mid X=a)P(X=a)}{\sum_{x}P(Y=b \mid X=x)P(X=x)}$$ | $$f_{X \mid Y} (a \mid b) = \frac{f_{X \mid Y}(b \mid a)f_{X}(a)}{\int_{-\infty}^{\infty}f_{X \mid Y}(b \mid x)f_{X}(x)dx}$$ |
| 독립성의 다른 표현 | 1. $$P(Y=b \mid X=a)$$가 $$a$$와 상관없다. <br> 2. $$P(Y=b \mid X=a) = P(Y=b)$$ <br>3. $$P(X=a, Y = 여러 가지)$$의 비가 $$a$$에 관계없이 일정하다.<br>4. $$P(X=a, Y=b) = P(X=a)P(Y=b)$$<br>5. $$P(X=a,Y=b) = g(a)h(b)$$의 형태 | 1. $$f_{Y \mid X} (b \mid a)$$가 $$a$$와 상관없다.<br>2. $$f_{Y \mid X} (b \mid a) = f_{Y}(b)$$<br>3. $$f_{X,Y} (a, 여러가지)$$의 비가 $$a$$에 관계없이 일정하다. <br>4. $$f_{X,Y}(a,b) = f_{X}(a)f_{Y}(b)$$ <br>5. $$f_{X,Y}(a,b) = g(a)h(b)$$의 형태 |



<h2>확률변수의 독립성</h2>

이산값과 유사하다.


$$
f_{Y \mid X}(b \mid a) = f_{Y}(b)
$$



가 항상 어떠한 a와 b에서도 성립할 때, X와 Y는 독립이다. 또한 아래도 성립한다.


$$
f_{Y \mid X}(b \mid a) = \frac{f_{X,Y}(a,b)}{f_{X}(a)} \\
\Rightarrow f_{X,Y}(a,b) = f_{X}(a)f_{Y}(b) \\
$$


<h2>결합분포의 변수변환</h2>

확률밀도함수의 변환 공식은 다음과 같다.



$$
f_{Z,W}(z,w) = \frac{1}{\begin{vmatrix} \partial (z,w) / \partial (x,y)\end{vmatrix}}f_{X,Y}(x,y) \ \ \ 단, z=g(x,y) , w=h(x,y)
$$



여기서 분모 부분은 야코비안이며 다음과 같이 정의된다.



$$
\frac{\partial (z,w)}{\partial(x,y)} \equiv det \begin{pmatrix} \frac{\partial z}{\partial x} & \frac{\partial z}{\partial y} \\ \frac{\partial w}{\partial x} & \frac{\partial w}{\partial y}\end{pmatrix}
$$

<h4>ex)</h4>

Q. X,Y의 결합분포의 확률밀도 함수를 $$f_{X,Y}(x,y)$$라 하고, $$Z \equiv 2X e^{X-Y}$$와 $$W \equiv X-Y$$의 결합분포의 확률밀도함수를 $$f_{Z,W}(z,w)$$라 할때, $$f_{Z,W}(6,0)$$을 $$f_{X,Y}$$로 나타내라.



A.

먼저, X와 Y는 다음과 같이 나타낼 수 있다.



$$
X=\frac{Z}{2e^{-W}} , Y=\frac{W}{2e^{-W}}-W
$$



이제 야코비안을 구한다.


$$
\begin{pmatrix} Z \\ W\end{pmatrix} = \begin{pmatrix} \frac{\partial z}{\partial x} & \frac{\partial z}{\partial y} \\ \frac{\partial w}{\partial x} & \frac{\partial w}{\partial y}\end{pmatrix} \begin{pmatrix} X \\Y \end{pmatrix} \\
\Rightarrow \begin{pmatrix} Z \\ W\end{pmatrix} = \begin{pmatrix} 2(e^{x-y}+xe^{x-y}) & -2xe^{x-y} \\1 & -1\end{pmatrix} \begin{pmatrix} X \\Y \end{pmatrix}
$$



그 다음 야코비안의 determinant를 구한다.



$$
\begin{vmatrix} 2(e^{x-y}+xe^{x-y}) & -2xe^{x-y} \\1 & -1 \end{vmatrix} = -2e^{x-y}-2xe^{x-y} + 2xe^{x-y} = -2e^{x-y}
$$



이제 이를 식에 대입하면



$$
f_{Z,W} (z,w) = \frac{1}{\begin{vmatrix}-2e^{x-y} \end{vmatrix}}f_{X,Y}(\frac{z}{2e^{-w}},\frac{z}{2e^{-w}}-w)
$$



마지막으로, z=6,w=0일 때, x와 y는 각각 3,3 이다. 따라서 식은 다음과 같이 완성된다.



$$
f_{Z,W}(6,0)=\frac{1}{\begin{vmatrix}-2e^{3-3}\end{vmatrix}}f_{X,Y}(3,3)=\frac{1}{2} f_{X,Y}(3,3)
$$


