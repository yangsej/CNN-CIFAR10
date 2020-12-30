# CNN-CIFAR10
Convolution Neural Network를 이용해 CIFAR10을 분류한다.  
  
## CNN(Convolution Neural Network)
기존에 사용하던 FCN(Fully Connected NetWork)의 단점을 극복하고자 선택된 방법이다.  


1. 입력에 raw data 대신에, clever feature extraction algorithm을 적용한다.  
2. Neural Network 대신에, SVM (Support Vector Machines), Random Forest Decision, 혹은 NN과 함께 적용한다.

### 참고) FCN의 단점  
1. 네트워크가 Deep 하게 들어갈수록 Error Propagation이 사라진다.  
2. 배워야 할 파라미터가 너무 많다. 

### CNN의 기초 연산  
1. Convolution  
  Raw Data에 Window형 Filter를 적용해 새로운 Feature Map을 계산한다.  
  Connection이 Sparse 하고, Weights도 서로 Shared 된다.  
  
2. Padding  
  Raw Data의 끝부분에 Filter의 중심을 적용할 경우 크기를 벗어나게 된다.  
  끝부분에 추가적인 테두리를 부여함으로써 이 문제를 해결한다.  
  
3. Strided Convolution  
  Filter를 몇칸씩 건너 뛰면서 적용한다.  
  
4. Dilated Convolution  
  Filter의 값들을 몇칸씩 건너 뛰어 Filter를 생성한다.   
  




 
  
## CIFAR10  
10개의 Class, 5만장의 training dataset, 1만장의 test data로 구성되어 있는 데이터셋

