%------------------------------
%Redes neuronais
%Ana Ferreira - al69136
%Diana Alves - al68557
%2020/2021
%------------------------------
clc
clear all
close all

n_epocas=2000;  % N� de epocas
alpha = 0.9;    %Fator de aprendizagem
SSE=zeros(1, n_epocas); %vetor soma dos erros quadr�ticos
N=4;    %N�mero de amostras
%Amostras de entrada fun��o XOR
X=[0 0 1;
   0 1 1;
   1 0 1;
   1 1 1 ];
%Sa�das da fun��o XOR
T= [ 0
     1
     1
     0];
 %Inicializa��o aleat�ria dos pesos [-1,1]
 %W1-2x3 w2-1x3
 W1 = 2*rand(2,3)*(2-1)
 W2 = 2*rand(1,3)*(2-1)
 
 for epoca = 1:n_epocas
     sum_sq_error = 0;
     for k=1:N
         x = X(k,:)';
         t= T(k);
         
         %Soma da camada de entrada
         g1 = W1*x;
         %Fun��o de ativa��o sigmoidal
         y1 = sig(g1);
         
         %Adi��o � sa�da da camada escondida y1 da entrada de bias com +1
         %Resulta em y1_b
         y1_b = [y1
                 1];
             
         %Soma da camada da sa�da 
         g2 = W2*y1_b;
         %Fun��o de ativa��o sigmoidal
         y2 = sig(g2);
         
         %Erro da camada de sa�da
         e= t-y2;
         
         %C�lculo do delta da camada Sigmoide
         delta2 = y2.*(1-y2).*e;
         
         %Atualiza��o da som dos erros quadr�ticos
         sum_sq_error = sum_sq_error + e^2;
         
         %Erro da camada escondida
         e1=W2'*delta2;
         %Erro sem o bias
         e1_b=e1(1:2);
         
         %C�lculo do delta da camada de sa�da
         delta1=y1.*(1-y1).*e1_b;
         
         %Atualiza��o dos pesos da camada escondida
         dW2=alpha*delta2*y1_b'; %com bias
         W2 = W2 +dW2;
         
         %Atualiza��o dos pesos da camada de entrada
         dW1 = alpha*delta1*x';
         W1 = W1 + dW1;
         
         %Sa�da prevista XOR
         y_plot(k) = sig(g2);
     end
     SSE(epoca) = (sum_sq_error)/N;
     y_plot
 end
 
 It = 1: 1: n_epocas;
 plot(It, SSE, 'r-', 'LineWidth', 2)
 xlabel('�poca')
 ylabel('SSE')
 title('Fun��o de ativa��o: Sigm�ide')
 

