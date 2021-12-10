%Author : Mahmut Ağralı
%Code is used for find the metrics for DQN algorithm

Tf=10;
Ts=0.01;

%load system
sAgent = 'DQN_savedAgents_v2/Agent646';
load(sAgent);

open_system('v_5_IHA_NEW_Plant/Scope/Scope_Z');
ScopeData = sim('v_5_IHA_NEW_Plant');

%get scope data
Data_Z=ScopeData.ScopeData_Z_DQN{1}.Values.Data;
size_data = size(Data_Z,3);
DQN_sig = zeros(size_data,1);
ref = zeros(size_data,1);
for i=1:size_data
    tmp_data = Data_Z(:,:,i);
    DQN_sig(i) = tmp_data(1);
    ref(i) = tmp_data(2);
end

% DQN_sig = reshape(ScopeData.ScopeData{1}.Values.Data,1,1001);
% ref = reshape(ScopeData.ScopeData{2}.Values.Data,1,1001);
error = ref- DQN_sig;
MSE = mean(error.^2);
ISE = sum(error.^2);
IAE = sum(abs(error));

fprintf("MSE : "+MSE+" ISE : "+ISE+" IAE : "+IAE + " at "+sAgent);