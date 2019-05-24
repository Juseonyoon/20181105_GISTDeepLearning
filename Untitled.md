
<2019-1 Deep Learning Final Report>

# <center>Realtime Recognition of Musical Directions Using Skeleton Data</center>  

20181105 기계공학부 윤주선  
Advisor 이규빈 교수님

※ 설명과 코드 재현을 위해 Jupyter와 Markdown으로 제작하여 제출합니다. 

# 1. Project Overview
___

<img src=image.png width=600 height=500>

## 1.1 Purpose

* To Recognize gesture for musical directions automatically in realtime

## 1.2 Necessity

* Robot musical instruments can follow the music conductor automatically.

* The conductor can be focused by camera automatically during the concert depending on the musical direction.

* For the educational purposes : for self checking, evaluation of the performance, or etc.

## 1.3 What to Recognize
* Start/End Gesture
* Rhythms (2/4, 3/4, 4/4, 6/8)
* Types (normal/staccato)
* Total 10 Gesture. 

<img src=image2.jpg width=600 height=500>

## 1.4 How to Develop

a. Microsoft 사의 Kinect V2 device를 사용하여 관절 데이터를 추출함. 관절 데이터는 이미지의 좌표로 총 24개의 값을 가지고 있음.  

b. Tensorflow 의 Recurent Neural Net 모듈을 사용하여 연속된 동작을 학습함 

c. Realtime이 가능하도록 weight와 array 최적화 

<img src=image1.png width=600 height=500>


## 1. Theoretical Basics - Recurent Neural Network 정의
___

일반적인 신경망은 각각의 입출력이 서로 독립적이라 가정하지만 RNN은 순서가 있는 정보를 입력 데이터로 사용한다는 점이 특징이다. 모든 입력 시퀀스에 대해 동일한 연산을 수행하지만 연산 시점에서의 출력 데이터는 이전의 모든 연산에 영향을 받는 특성이 있다.

RNNs은 배열에 등장했던 패턴을 ‘기억’할 수 있는 능력이 있다. 과거의 출력이 다시 입력이 되는 구조를 소위 피드백 구조라고 한다. 피드백 구조는 방금 일어난 출력을 다시 입력으로 집어 넣는데, 이 구조 덕분에 RNNs은 기억 능력이 있다고 한다.

<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/1024px-Recurrent_neural_network_unfold.svg.png >

RNN의 기본 구조는 위 그림과 같다. 파란 박스는 히든 state를 의미한다. 초록 박스는 인풋 $x$, 빨간 박스는 아웃풋 $o$이다. 현재 상태의 히든 state $h_t$는 직전 시점의 히든 state $h_{t−1}$를 받아 갱신된다. 은닉층의 메모리 셀은 $h_t$를 계산하기 위해서 총 두 개의 가중치를 갖게 된다. 하나는 입력층에서 입력값을 위한 가중치 $V$이고, 하나는 이전 시점 t-1의 은닉 상태값인  $h_{t−1}$을 위한 가중치 $U$이다.

> * 은닉층 : $h_{t} = tanh(V * x_{t} + U * h_{t−1} + b)$
> * 출력층 : $y_{t} = f(W * h_{t} + b)$

$V, U, W$로 명명된 <u>모두 같은 weight를 공유</u> 한다. 따라서 학습 대상 weight의 갯수가 크게 감소해 학습 시간이 감소한다. 이때 $h_{t}$를 계산하기 위한 활성화 함수로는 주로 하이퍼볼릭탄젠트 함수($tanh$)가 쓰인다. ReLU도 사용할 수는 있지만, 대부분의 딥 러닝 연구자들은 하이퍼볼릭탄젠트 함수를 주로 사용한다. 첫 은닉 상태인 $h_1$을 계산 하기 위해서는 이전 은닉 상태가 존재하지 않으므로 $h_0$의 값을 임의로 정해줘야하는데 주로 0을 사용한다. 

출력층은 결과값인 $y_t$를 계산하기 위한 활성화 함수로는 상황에 따라 다를텐데, 예를 들어서 이진 분류를 해야하는 경우라면 시그모이드 함수를 사용할 수 있고 다양한 카테고리 중에서 선택해야하는 문제라면 소프트맥스 함수를 사용한다.

위의 식에서 또 주목해야할 점은 각각의 가중치 $V, U, W$는 시점에 따라서 전혀 변화하지 않는다는 점이다. 즉, RNN의 (하나의 은닉층 내에서) 모든 시점에서 가중치 $V, U, W$는 동일하게 공유한다. 만약, 은닉층이 2개 이상일 경우에는 은닉층 2개의 가중치는 서로 다르다.

<div class="alert alert-info"><h4>Note</h4><p> RNN에서 하이퍼볼릭 탄젠트($tanh$)를 activation function으로 주로 사용하는 이유는?  
    
* Gradient Vanishing 문제를 해결하기위해 2차미분을 유지하는 함수가 필요함. 
* Error의 흐름을 양수와 음수로 제어하여 상태의 증가나 감소가 가능함. sigmoid의 경우 증가만 가능함. (y축의 범위가$tanh$는 [-1 1], $sigmoid$는 [0 1])
* dTanh(x)/dx 는 최대값이 1기 때문에 sigmoid(0.25) 보다 gradient vanishing에 강함.

</p></div>

### ※ Feed Forward Neural Network(FFnet)

<img src=https://deeplearning4j.org/img/feedforward_rumelhart.png width="300">

FFnet는 입력층 --> 출력층 한방향으로만 흐른다.즉, 데이터를 입력하면 입력측에서 은닉층까지 연산이 순차적으로 진행되고 출력이 발생한다. 이 과정에서 RNN과는 다르게 노드를 한번만 지나간다. 데이터가 노드를 한번 지나간다는 것은 데이터의 순서, **시간**을 고려하지 않는 구조라는 뜻이다. FFnets 에서 여러개의 은닉층을 가질경우 **Deep Neural Network**라고 한다. FFnets를 확장하여 수학적 개념인 convolution을 적용할 경우 공간적 개념을 포함 할 수 있다. 이를 **Convolutional Neural Network**라 한다. 

반대로 RNN은 은닉층의 결과가 다시 같은 은닉층의 입력으로 들어간다. 이는 순서 혹은 시간의 개념을 포함 할 수 있는 특징을 가지게 된다. 

## 1.1 RNN 종류


## 1.2 RNN Application
### 1.1.1 Natural Language Processing
### 1.1.2 Speech Recognition
### 1.1.3 Signal Processing
### 1.1.4 Image Recognition & Characterization
### 1.1.5 Image Captioning
### 1.1.6 Etc

## 1.3 RNN 단점
### - Gradient Vanishing
https://skymind.ai/kr/wiki/lstm
https://aikorea.org/blog/rnn-tutorial-3/
### - Backpropagation Through Time (BPTT)
https://aikorea.org/blog/rnn-tutorial-3/


※

# 2. LSTM
___





# 3. Project Recap
___
## 3.1 Datasets

<img src=image3.png >


# 4. 코드로 구현하기
## 환경









#  References

Feed forward vs RNN

- Ref. https://skymind.ai/kr/wiki/lstm


Basic RNN
- Ref. http://jaejunyoo.blogspot.com/2017/06/anyone-can-learn-to-code-LSTM-RNN-Python.html
- Ref. https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/
- Ref. https://ratsgo.github.io/deep%20learning/2017/05/14/backprop/
- Ref. https://wikidocs.net/22886
- Ref. https://wegonnamakeit.tistory.com/7
Many to one, one to many, many to many
- Ref. 
- Ref. 

LSTM
