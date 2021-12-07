%------------------------------
%Redes neuronais
%Ana Ferreira - al69136
%Diana Alves - al68557
%2020/2021
%------------------------------
clc
clc
clear all
close all

Data_multi
K=40;
%Classe_A=[1 0 0];
%Classe_B=[0 1 0];
%Classe_C=[0 0 1];
A=X_A';
B=X_B';
C=X_C';

plot(A(1,:),A(2,:),'bs')
grid on
hold on
plot(B(1,:),B(2,:),'r+')
plot(C(1,:),C(2,:),'ko')
hold off

W1 = 2*rand(4, 3)*(2-1);
W2 = 2*rand(3, 5)*(2-1);

bias_A= ones(K, 1);
X_A=[X_A,bias_A];
T_A=ones(K,3).*[1 0 0];

bias_B= ones(K, 1);
X_B=[X_B,bias_B];
T_B=ones(K,3).*[0 1 0];

bias_C= ones(K, 1);
X_C=[X_C,bias_C];
T_C=ones(K,3).*[0 0 1];
X = [ X_A;
      X_B;
      X_C]
T = [ T_A;
      T_B;
      T_C]


n_epochs = 10000; %numero de epocas
alpha = 0.9;   %fator de aprendizagem
%vetor soma dos erros quadraticos
SSE = zeros(1,n_epochs);
N = 3*K;           %numero de amostras
%amostras de entrada funcao XOR

%inicializacao aleatoria dos pesos [-1, 1]
%W1-2x3 W2-1x3
% W1 = 2*rand(2,3) - 1
% W2 = 2*rand(1,3) - 1

for epoch = 1:n_epochs
    sum_sq_error=0;
    for k = 1:N
        x = X(k,:)';
        t = T(k,:)';
        %soma da camada de entrada
        g1 = W1*x;
        %funcao de ativacao sigmoidal
        y1 = sig(g1);
        %adicao a saida da camada escondida y1
        %da entrada de bias com +1
        %resulta em y1_b
        y1_b = [y1
                1];
        %soma da camada de saida
        g2 = W2*y1_b;
        %funcao de ativacao sigmoidal
        y2 = sig(g2);
        %erro da camada de saida
        e = t-y2;
        %calculo do delta da camada de saida
        %sigmoide
        delta2 = y2.*(1-y2).*e;
        %atualizacao da soma dos erros quadraticos
        sum_sq_error = sum_sq_error + sum(e.^2);
        %erro da camada escondida
        e1 = W2'*delta2;
        %erro sem o bias
        e1_b = e1(1:4);
        %calculo do delta da camada de saida
        delta1 = y1.*(1-y1).*e1_b;
        %atualizacao dos pesos da camada escondida
        dW2 = alpha*delta2*y1_b';
        W2 = W2 + dW2;
        %atualizacao dos pesos da camada de entrada
        dW1 = alpha*delta1*x';
        W1 = W1 + dW1;
    end
     SSE(epoch) = (sum_sq_error)/N;
end
y_plot=zeros(N:3);
for k = 1:N
    x = X(k,:)';
    g1 = W1*x;
    %sigmoide
    y1 = sig(g1);
    %y1 mais uma entrada de bias
    y1_b = [y1
            1];
    g2 = W2*y1_b;
    y2 = sig(g2);
    %saida prevista XOR
    y_plot(k,:) = y2;
end
y_plot

It = 1:1:n_epochs;
semilogx(It,SSE,'r-','LineWidth',2)
xlabel('Epoca')
ylabel('SSE')
title('Funcao de ativacao: sigmoide')

function [s] = sig(x)
    s = 1./(1+exp(-x));
end