%Author : Mahmut Ağralı
%Code is used for find the metrics for PD algorithm

%load system
open_system('v_5_IHA_NEW_Plant/Scope/Scope_Z');
ScopeData = sim('v_5_IHA_NEW_Plant');

%get scope data
Data_Z=ScopeData.ScopeData_Z_PD{1}.Values.Data;
size_data = size(Data_Z,3);
PD_sig = zeros(size_data,1);
ref = zeros(size_data,1);
for i=1:size_data
    tmp_data = Data_Z(:,:,i);
    PD_sig(i) = tmp_data(1);
    ref(i) = tmp_data(2);
end

%find metrics
error = ref - PD_sig;
MSE = mean(error.^2);
ISE = sum(error.^2);
IAE = sum(abs(error));

fprintf("MSE : "+MSE+" ISE : "+ISE+" IAE : "+IAE + " at P : 17.9012745706573, I : 22.4780602564623, D : 3.11340256685474");