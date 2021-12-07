%------------------------------
%Redes neuronais
%Ana Ferreira - al69136
%Diana Alves - al68557
%2020/2021
%------------------------------
clc
clear all
close all

n_epocas=2000;  % Nº de epocas
alpha = 0.9;    %Fator de aprendizagem
SSE=zeros(1, n_epocas); %vetor soma dos erros quadráticos
N=4;    %Número de amostras
%Amostras de entrada função XOR
X=[0 0 1;
   0 1 1;
   1 0 1;
   1 1 1 ];
%Saídas da função XOR
T= [ 0
     1
     1
     0];
 %Inicialização aleatória dos pesos [-1,1]
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
         %Função de ativação sigmoidal
         y1 = sig(g1);
         
         %Adição à saída da camada escondida y1 da entrada de bias com +1
         %Resulta em y1_b
         y1_b = [y1
                 1];
             
         %Soma da camada da saída 
         g2 = W2*y1_b;
         %Função de ativação sigmoidal
         y2 = sig(g2);
         
         %Erro da camada de saída
         e= t-y2;
         
         %Cálculo do delta da camada Sigmoide
         delta2 = y2.*(1-y2).*e;
         
         %Atualização da som dos erros quadráticos
         sum_sq_error = sum_sq_error + e^2;
         
         %Erro da camada escondida
         e1=W2'*delta2;
         %Erro sem o bias
         e1_b=e1(1:2);
         
         %Cálculo do delta da camada de saída
         delta1=y1.*(1-y1).*e1_b;
         
         %Atualização dos pesos da camada escondida
         dW2=alpha*delta2*y1_b'; %com bias
         W2 = W2 +dW2;
         
         %Atualização dos pesos da camada de entrada
         dW1 = alpha*delta1*x';
         W1 = W1 + dW1;
         
         %Saída prevista XOR
         y_plot(k) = sig(g2);
     end
     SSE(epoca) = (sum_sq_error)/N;
     y_plot
 end
 
 It = 1: 1: n_epocas;
 plot(It, SSE, 'r-', 'LineWidth', 2)
 xlabel('Época')
 ylabel('SSE')
 title('Função de ativação: Sigmóide')
 

