%Author : Mahmut Ağralı
%Code is used to find compare plot for all algorithms

%for compare all algorithms
figure
hold on

plot(0:0.01:10,ref(1:1001),'black','LineWidth',2)
plot(0:0.01:10,DQN_sig(1:1001),'blue','LineWidth',2)
% plot(0:0.01:10,DDPG_sig(1:1001),'red','LineWidth',2)
plot(0:0.01:10,PD_sig(1:1001),'red','LineWidth',2)

legend('Reference','DQN','PD')
xlabel({'Time(s)'})
ylabel({'Z- Altitude(m)'})
%title({'The step response of the algorithms'})
hold off

%% For Phi Angle 
Data_Phi=ScopeData.ScopeData_Phi{1}.Values.Data;
size_data = size(Data_Phi,3);
PD_Phi_sig = zeros(size_data,1);
ref_PD = zeros(size_data,1);

for i=1:size_data
    tmp_data = Data_Phi(:,:,i);
    PD_Phi_sig(i) = tmp_data(1);
    ref_PD(i) = tmp_data(2);
end

figure
hold on

plot(0:0.01:10,ref_PD(1:1001),'black','LineWidth',2)
plot(0:0.01:10,PD_Phi_sig(1:1001),'blue','LineWidth',2)

legend('Reference','PD')
xlabel({'Time(s)'})
ylabel({'Phi Angle(rad)'})
%title({'The step response of the algorithms'})
hold off

%% For Theta Angle 
Data_Theta=ScopeData.ScopeData_Theta{1}.Values.Data;
size_data = size(Data_Theta,3);
PD_Theta_sig = zeros(size_data,1);
ref_PD = zeros(size_data,1);

for i=1:size_data
    tmp_data = Data_Theta(:,:,i);
    PD_Theta_sig(i) = tmp_data(1);
    ref_PD(i) = tmp_data(2);
end

figure
hold on

plot(0:0.01:10,ref_PD(1:1001),'black','LineWidth',2)
plot(0:0.01:10,PD_Theta_sig(1:1001),'blue','LineWidth',2)

legend('Reference','PD')
xlabel({'Time(s)'})
ylabel({'Theta Angle(rad)'})
%title({'The step response of the algorithms'})
hold off

%% For Psi Angle 
Data_Psi=ScopeData.ScopeData_Theta{1}.Values.Data;
size_data = size(Data_Psi,3);
PD_Psi_sig = zeros(size_data,1);
ref_PD = zeros(size_data,1);

for i=1:size_data
    tmp_data = Data_Psi(:,:,i);
    PD_Psi_sig(i) = tmp_data(1);
    ref_PD(i) = tmp_data(2);
end

figure
hold on

plot(0:0.01:10,ref_PD(1:1001),'black','LineWidth',2)
plot(0:0.01:10,PD_Psi_sig(1:1001),'blue','LineWidth',2)

legend('Reference','PD')
xlabel({'Time(s)'})
ylabel({'Psi Angle(rad)'})
%title({'The step response of the algorithms'})
hold off
