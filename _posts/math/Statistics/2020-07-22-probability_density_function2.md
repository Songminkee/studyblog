---
title: 확률밀도의 결합분포, 주변분포, 조건부분포, 기대값, 분산, 표준편차
author: Monch
category: Statistics
layout: post
---

<h2>결합 분포</h2>

확률 변수 X,Y에 대해 $$W \equiv (X,Y)$$이 정의 될 때, W를 X,Y의 결합분포라고 부른다.

<br>

<h2>이산값(확률) VS 실수값(확률밀도)</h2>

| $$\ \ \ \ \ \ \ \ $$ |                         이산값(확률)                         |                       실수값(확률밀도)                       |
| -------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 주변분포             |               $$P(X=a) = \sum_{y} P(X=a,Y=y)$$               |     $$f_{X}(a) = \int^{\infty}_{-\infty}f_{X,Y}(a,y)dy$$     |
| 조건부분포           | $$P(Y=b \mid X=a) \equiv \frac{P(X=a,Y=b)}{P(X=a)} \\ P(X=a,Y=b) = P(Y=b \mid X=a) P(X=a)$$ | $$f_{Y \mid X}(b \mid a) \equiv \frac{f_{X,Y}(a,b)}{f_{X}(a)} \\ f_{X,Y}(a,b) = f_{Y \mid X} (b \mid a)f_{X}(a)$$ |
| 베이즈 공식          | $$P(X=a \mid Y=b) = \frac{P(Y=b \mid X=a)P(X=a)}{\sum_{x}P(Y=b \mid X=x)P(X=x)}$$ | $$f_{X \mid Y} (a \mid b) = \frac{f_{X \mid Y}(b \mid a)f_{X}(a)}{\int_{-\infty}^{\infty}f_{X \mid Y}(b \mid x)f_{X}(x)dx}$$ |
| 독립성의 다른 표현   | 1. $$P(Y=b \mid X=a)$$가 $$a$$와 상관없다. <br> 2. $$P(Y=b \mid X=a) = P(Y=b)$$ <br>3. $$P(X=a, Y = 여러 가지)$$의 비가 $$a$$에 관계없이 일정하다.<br>4. $$P(X=a, Y=b) = P(X=a)P(Y=b)$$<br>5. $$P(X=a,Y=b) = g(a)h(b)$$의 형태 | 1. $$f_{Y \mid X} (b \mid a)$$가 $$a$$와 상관없다.<br>2. $$f_{Y \mid X} (b \mid a) = f_{Y}(b)$$<br>3. $$f_{X,Y} (a, 여러가지)$$의 비가 $$a$$에 관계없이 일정하다. <br>4. $$f_{X,Y}(a,b) = f_{X}(a)f_{Y}(b)$$ <br>5. $$f_{X,Y}(a,b) = g(a)h(b)$$의 형태 |
| 기댓값               | 1. $$E[X] \equiv X(w) 그래프의 \ 부피$$ <br>2. $$E[X] = \sum_{x} xP(X=x)$$<br>3. $$E[g(X)] = \sum_{x}g(x)P(X=x)$$<br>4. $$E[h(X,Y)] = \sum_{y} \sum_{x} h(x,y) P(X=x,Y=y)$$<br>5. $$E[aX+b] = aE[X] +b$$ | 1. 동일<br>2. $$E[X] = \int^{\infty}_{-\infty} xf_{X}(x)dx$$<br>3. $$E[g(X)] = \int^{\infty}_{-\infty} g(x)f_{X}(x)dx$$<br>4. $$E[h(X,Y)] = \int^{\infty}_{-\infty}h(x,y)f_{X,Y}(x,y)dxdy$$<br> |
| 분산                 | 1. $$V[X] \equiv E[(X-\mu)^2], \ \ \mu \equiv E[X]$$<br>2. $$V[aX+b]=a^2 V[X]$$<br> |                      1. 동일<br>2. 동일                      |
| 표준편차             | 1. $$\sigma_{X} \equiv \sqrt{V[X]}$$<br>2. $$\sigma_{aX+b} = \begin{vmatrix}a\end{vmatrix} \sigma_{X}$$<br> |                    1. 동일<br>2. 동일<br>                    |
| 조건부 기댓값        |    1. $$E[Y \mid X = a] \equiv \sum_{b}bP(Y=b \mid X=a)$$    | 1. $$E[Y \mid X =a] \equiv \int^{\infty}_{-\infty} yf_{Y}(y \mid X=a) dy$$ |
| 조건부 분산          |    1. $$V[Y \mid X =a] \equiv E[(Y-\mu(a))^2 \mid X=a]$$     |                             동일                             |

<br>

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

<br>

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



<h2>확률밀도의 기댓값, 분산, 표준편차 예제</h2>

<h4>ex1)</h4>

Q1. 확률변수 $$X$$의 확률밀도함수 f_{X}(x)가 다음 식으로 주어질 때 $$E[X]$$와 $$E[X^2]$$의 값


$$
f_{X}(x) = \begin{cases}2x \ (0 \le x \le 1) \\ 0 \ \ \ (기타) \end{cases}
$$


A1.


$$
E[X] = \int^{\infty}_{-\infty} xf_{X}(x)dx = \int^{1}_{0}x(2x)dx = \left[\frac{2}{3}x^{3} \right]^{1}_{0} = \frac{2}{3} \\
E[X^2] = E[g(X)] = \int^{\infty}_{-\infty} g(x)f_{X}(x)dx = \int^{1}_{0}x^2(2x)dx = \left[ \frac{2}{4}x^{4} \right]^{1}_{0} = \frac{1}{2}
$$
<br>

<br>

<br>

Q2. 다음의 성질을 적분 계산으로부터 유도하라

- $$E[3X] = 3E[X]$$.
- $$E[X+3] = E[X]+3 $$.



A2.


$$
E[3X]=\int^{\infty}_{-\infty}3xf_{X}(x)dx =3\int^{\infty}_{-\infty}xf_{X}(x)dx=3E[X] \\
E[X+3]=\int^{\infty}_{-\infty}(x+3)f_{X}(x)dx = \int^{\infty}_{-\infty}xf_{X}(x)dx+3\int^{\infty}_{-\infty}f_{X}(dx)dx = E[X]+3\times1=E[X]+3
$$



<br>

<br>

<br>


Q3. 확률변수 $$X$$의 확률밀도함수 $$f_{X}(x)가 다음 식으로 주어질 때 분산 $$V[X]$$와 표준편차 $$\sigma$$를 구하시오.


$$
f_{X}(x)= \begin{cases}2x \ (0 \le x \le 1) \\ 0 \ \ \ (기타) \end{cases}
$$


A3.



위에서 $$E[X]=2/3$$을 구했으므로


$$
V[X]=E \left[ \left(X-\frac{2}{3} \right)^2\right] = \int^{\infty}_{-\infty} \left(x-\frac{2}{3} \right)^{2} f_{X}(x)dx = \int^{1}_{0} \left(x-\frac{2}{3} \right)^{2} (2x)dx \\
\int^{1}_{0} \left( 2x^{3} - \frac{8}{3}x^{2} + \frac{8}{9}x \right)dx = \left[\frac{1}{2}x^{4}-\frac{8}{9}x^{3}+\frac{4}{9}x^{2} \right]^{1}_{0} = \frac{1}{2}-\frac{8}{9}+\frac{4}{9} = \frac{1}{18} \\
$$


혹은 분산의 성질에 의해




$$
V[X]=E[(X- \mu)^2] \\ =  E[X^{2}-2X \mu + \mu^2] \\ =  E[X^2]-2E[X]E[X] + E[X]^2 \\ = E[X^2]-E[X]^2
$$


이므로




$$
V[X]=E[X^2]-E[X]^2 = \frac{1}{2} - \frac{4}{9} = \frac{1}{18}
$$
로 구할 수 있다.


$$
\sigma =\sqrt{V[X]} = \sqrt{\frac{1}{18}}=\frac{1}{3 \sqrt{2}}
$$
