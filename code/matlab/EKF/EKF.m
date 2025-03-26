%% ������ʹ��EKF����﮵��SOC
clear;clc;
%% ģ�Ͳ���
load('R0.mat');
load('R1.mat');
R1=R1+0.0007; %��Ϊ�������
load('R2.mat');
R2=R2+0.0005; %��Ϊ�������
load('C1.mat');
load('C2.mat');
load('discharge.mat');%�ŵ�����
% discharge=discharge(:,1:250000);
load('OCV_SOC.mat');%OCV-SOC��ϵ
Qn=30.23*3600;%��������λ��A*S
Ts=0.1;%�������
%% ����
a1 = exp(-Ts/(R1*C1));
a2 = exp(-Ts/(R2*C2));
% a1 = 1 - Ts/(R1*C1);
% a2 = 1 - Ts/(R2*C2);
A=[a1 0 0;0 a2 0;0 0 1];%ϵͳ����
b1 = R1 * (1-a1);
b2 = R2 * (1-a2);
B=[b1; b2; -Ts/Qn];
C=[-1 -1 0];
D= -R0;
%% ��ʼֵ
Q=0.0000001*eye(3);%��Ϊ�����������
R=0.1;%�������Э����
P0=[0.01 0 0;0 0.01 0;0 0 0.1];%״̬���Э�����ʼֵ
%% ��ֵ
tm=discharge(1,:)';%ʱ��
Cur=-discharge(2,:)';%����
Vot=discharge(3,:)';%�����õ��Ķ˵�ѹ
RSOC=discharge(4,:)';%SOC��ʵֵ-��ʱ������õ�
T=length(tm)-1;%ʱ��
%% ocv-soc��ϵ
x=OCV_SOC(2,:);%SOC
y=OCV_SOC(1,:);%OCV
p=polyfit(x,y,8);%����ʽ����ֵ
%% EKF�㷨����SOC
Xekf=[0;0;0.8];%[U1,U2,SOC]��ʼֵ
Uoc=zeros(1,T+1);%OCV
Vekf=zeros(1,T+1);%���Ƶõ��Ķ˵�ѹֵ
% Uoc(1)=p(1)*Xekf(3)^8+p(2)*Xekf(3)^7+p(3)*Xekf(3)^6+p(4)*Xekf(3)^5+p(5)*Xekf(3)^4+p(6)*Xekf(3)^3+p(7)*Xekf(3)^2+p(8)*Xekf(3)+p(9);%OCV
Uoc(1)=polyval(p,Xekf(3));
Vekf(1)=Uoc(1)+C*Xekf-Cur(1)*R0;%���Ƶõ��Ķ˵�ѹֵ
K=zeros(3,T);%����������
H=zeros(T,3);%�۲����
for i=1:T
    Xekf(:,i+1)=A*Xekf(:,i)+B*Cur(i);%����״ֵ̬
%     Uoc(i+1)=p(1)*Xekf(3,i+1)^8+p(2)*Xekf(3,i+1)^7+p(3)*Xekf(3,i+1)^6+p(4)*Xekf(3,i+1)^5+p(5)*Xekf(3,i+1)^4+p(6)*Xekf(3,i+1)^3+p(7)*Xekf(3,i+1)^2+p(8)*Xekf(3,i+1)+p(9);
    Uoc(i+1)=polyval(p, Xekf(3, i+1));
    dp=polyder(p);
    dOCV_dSOC=polyval(dp, Xekf(3,i+1));
%     H(i,:)=[-1 -1 p(1)*8*Xekf(3,i+1)^7+p(2)*7*Xekf(3,i+1)^6+p(3)*6*Xekf(3,i+1)^5+p(4)*5*Xekf(3,i+1)^4+p(5)*4*Xekf(3,i+1)^3+p(6)*3*Xekf(3,i+1)^2+p(7)*2*Xekf(3,i+1)+p(8)];
    H(i,:)=[-1 -1 dOCV_dSOC];
    Vekf(i+1)=Uoc(i+1)+C*Xekf(:,i+1) + Cur(i+1)* D;%���Ƶõ��Ķ˵�ѹֵ
    P=A*P0*A'+Q;%����״̬���Э����
    K(:,i)=P*H(i,:)'/(H(i,:)*P*H(i,:)'+R);%����������
    Xekf(:,i+1)=Xekf(:,i+1)+K(:,i)*(Vot(i+1)-Vekf(i+1));%����״ֵ̬
    P0=(eye(3)-K(:,i)*H(i,:))*P;%����״̬���Э����
end
%% ��ͼ
t=0:0.1:length(tm)/10-0.1;
figure(1);
plot(t,Vot,'-k',t,Vekf,'-r','lineWidth',2); grid on
legend('��ʵֵ','����ֵ-EKF');
ylabel('�˵�ѹ','Fontsize', 16)
xlabel('ʱ��(s)', 'Fontsize', 16)

figure(2);
plot(t,RSOC,'-k',t,Xekf(3,:),'-r','lineWidth',2); grid on
legend('��ʵֵ','����ֵ-EKF');
ylabel('SOC','Fontsize', 16)
xlabel('ʱ��(s)', 'Fontsize', 16)
V_error=Vot-Vekf';
SOC_error=RSOC-Xekf(3,:)';
SOC_error_mean=mean(abs(SOC_error(5000:end)));
SOC_error_max=max(abs(SOC_error(5000:end)));
SOC_error_rmse=sqrt(mean(SOC_error(5000:end).^2));
fprintf('SOC��RMSE: %.4f\n', SOC_error_rmse);
V_error_rmse = sqrt(mean(V_error(5000:end).^2)); % �˵�ѹ RMSE
fprintf('�˵�ѹ��������RMSE����%.4f\n', V_error_rmse);

figure(3);
plot(t,V_error,'-k','lineWidth',2); grid on
legend('�˵�ѹ��� ');
ylabel('�˵�ѹ���','Fontsize', 16)
xlabel('ʱ��(s)', 'Fontsize', 16)
figure(4);
plot(t,SOC_error,'-k','lineWidth',2); grid on
legend('SOC���');
ylabel('SOC���','Fontsize', 16)
xlabel('ʱ��(s)', 'Fontsize', 16)

SOC_EKF=Xekf(3,:);
save SOC_EKF.mat SOC_EKF
SOC_error_EKF=SOC_error;
save SOC_error_EKF.mat SOC_error_EKF
save RSOC.mat RSOC