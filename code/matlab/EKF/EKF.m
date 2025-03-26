%% 本程序使用EKF估计锂电池SOC
clear;clc;
%% 模型参数
load('R0.mat');
load('R1.mat');
R1=R1+0.0007; %人为增大误差
load('R2.mat');
R2=R2+0.0005; %人为增大误差
load('C1.mat');
load('C2.mat');
load('discharge.mat');%放电数据
% discharge=discharge(:,1:250000);
load('OCV_SOC.mat');%OCV-SOC关系
Qn=30.23*3600;%容量，单位：A*S
Ts=0.1;%采样间隔
%% 矩阵
a1 = exp(-Ts/(R1*C1));
a2 = exp(-Ts/(R2*C2));
% a1 = 1 - Ts/(R1*C1);
% a2 = 1 - Ts/(R2*C2);
A=[a1 0 0;0 a2 0;0 0 1];%系统矩阵
b1 = R1 * (1-a1);
b2 = R2 * (1-a2);
B=[b1; b2; -Ts/Qn];
C=[-1 -1 0];
D= -R0;
%% 初始值
Q=0.0000001*eye(3);%人为调整增大误差
R=0.1;%测量误差协方差
P0=[0.01 0 0;0 0.01 0;0 0 0.1];%状态误差协方差初始值
%% 赋值
tm=discharge(1,:)';%时间
Cur=-discharge(2,:)';%电流
Vot=discharge(3,:)';%测量得到的端电压
RSOC=discharge(4,:)';%SOC真实值-安时法计算得到
T=length(tm)-1;%时间
%% ocv-soc关系
x=OCV_SOC(2,:);%SOC
y=OCV_SOC(1,:);%OCV
p=polyfit(x,y,8);%多项式参数值
%% EKF算法估计SOC
Xekf=[0;0;0.8];%[U1,U2,SOC]初始值
Uoc=zeros(1,T+1);%OCV
Vekf=zeros(1,T+1);%估计得到的端电压值
% Uoc(1)=p(1)*Xekf(3)^8+p(2)*Xekf(3)^7+p(3)*Xekf(3)^6+p(4)*Xekf(3)^5+p(5)*Xekf(3)^4+p(6)*Xekf(3)^3+p(7)*Xekf(3)^2+p(8)*Xekf(3)+p(9);%OCV
Uoc(1)=polyval(p,Xekf(3));
Vekf(1)=Uoc(1)+C*Xekf-Cur(1)*R0;%估计得到的端电压值
K=zeros(3,T);%卡尔曼增益
H=zeros(T,3);%观测矩阵
for i=1:T
    Xekf(:,i+1)=A*Xekf(:,i)+B*Cur(i);%先验状态值
%     Uoc(i+1)=p(1)*Xekf(3,i+1)^8+p(2)*Xekf(3,i+1)^7+p(3)*Xekf(3,i+1)^6+p(4)*Xekf(3,i+1)^5+p(5)*Xekf(3,i+1)^4+p(6)*Xekf(3,i+1)^3+p(7)*Xekf(3,i+1)^2+p(8)*Xekf(3,i+1)+p(9);
    Uoc(i+1)=polyval(p, Xekf(3, i+1));
    dp=polyder(p);
    dOCV_dSOC=polyval(dp, Xekf(3,i+1));
%     H(i,:)=[-1 -1 p(1)*8*Xekf(3,i+1)^7+p(2)*7*Xekf(3,i+1)^6+p(3)*6*Xekf(3,i+1)^5+p(4)*5*Xekf(3,i+1)^4+p(5)*4*Xekf(3,i+1)^3+p(6)*3*Xekf(3,i+1)^2+p(7)*2*Xekf(3,i+1)+p(8)];
    H(i,:)=[-1 -1 dOCV_dSOC];
    Vekf(i+1)=Uoc(i+1)+C*Xekf(:,i+1) + Cur(i+1)* D;%估计得到的端电压值
    P=A*P0*A'+Q;%先验状态误差协方差
    K(:,i)=P*H(i,:)'/(H(i,:)*P*H(i,:)'+R);%卡尔曼增益
    Xekf(:,i+1)=Xekf(:,i+1)+K(:,i)*(Vot(i+1)-Vekf(i+1));%后验状态值
    P0=(eye(3)-K(:,i)*H(i,:))*P;%后验状态误差协方差
end
%% 画图
t=0:0.1:length(tm)/10-0.1;
figure(1);
plot(t,Vot,'-k',t,Vekf,'-r','lineWidth',2); grid on
legend('真实值','估计值-EKF');
ylabel('端电压','Fontsize', 16)
xlabel('时间(s)', 'Fontsize', 16)

figure(2);
plot(t,RSOC,'-k',t,Xekf(3,:),'-r','lineWidth',2); grid on
legend('真实值','估计值-EKF');
ylabel('SOC','Fontsize', 16)
xlabel('时间(s)', 'Fontsize', 16)
V_error=Vot-Vekf';
SOC_error=RSOC-Xekf(3,:)';
SOC_error_mean=mean(abs(SOC_error(5000:end)));
SOC_error_max=max(abs(SOC_error(5000:end)));
SOC_error_rmse=sqrt(mean(SOC_error(5000:end).^2));
fprintf('SOC—RMSE: %.4f\n', SOC_error_rmse);
V_error_rmse = sqrt(mean(V_error(5000:end).^2)); % 端电压 RMSE
fprintf('端电压均方根误差（RMSE）：%.4f\n', V_error_rmse);

figure(3);
plot(t,V_error,'-k','lineWidth',2); grid on
legend('端电压误差 ');
ylabel('端电压误差','Fontsize', 16)
xlabel('时间(s)', 'Fontsize', 16)
figure(4);
plot(t,SOC_error,'-k','lineWidth',2); grid on
legend('SOC误差');
ylabel('SOC误差','Fontsize', 16)
xlabel('时间(s)', 'Fontsize', 16)

SOC_EKF=Xekf(3,:);
save SOC_EKF.mat SOC_EKF
SOC_error_EKF=SOC_error;
save SOC_error_EKF.mat SOC_error_EKF
save RSOC.mat RSOC