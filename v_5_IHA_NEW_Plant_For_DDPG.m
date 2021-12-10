%Author : Mahmut Ağralı
%Code is used to train the DDPG agent

clear;clc;
mdl = 'v_5_IHA_NEW_Plant' ; 
open_system(mdl);

%load the system
open_system('v_5_IHA_NEW_Plant/Scope/Scope_Z');
open_system('v_5_IHA_NEW_Plant/Controller/Scope_PD_DQN_DDPG_Controller_Out');

%set parameters for observation 
obsInfo = rlNumericSpec([3 1],'LowerLimit',[-inf -inf -inf]','UpperLimit',[inf  inf inf]'); %[4 1],'LowerLimit',[-inf -inf -inf 0]','UpperLimit',[inf  inf inf inf]');
obsInfo.Name = 'observation';
obsInfo.Description = 'integrated error, error, and Actual Z';
numObservations = obsInfo.Dimension(1);

%set parameters for action
actInfo = rlNumericSpec([1 1],'LowerLimit',-2,'UpperLimit',2);
actInfo.Name = 'action';
actInfo.Description = 'Z Altitude';
numActions = actInfo.Dimension(1);

%call the simulink environment
env = rlSimulinkEnv('v_5_IHA_NEW_Plant','v_5_IHA_NEW_Plant/Controller/PD and DQN-DDPG Controller for Z/DQN-DDPG Controller for Z/RL Agent',obsInfo,actInfo);

%set reset function
env.ResetFcn = @(in)localResetFcn(in);

%set step size and max simulation time
Ts = 0.01;
Tf = 10;

%initilize randomness
rng(0)

%design state path
statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','observation')
    fullyConnectedLayer(400,'Name','CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(300,'Name','CriticStateFC2')];

%design action path
actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','action')
    fullyConnectedLayer(300,'Name','CriticActionFC1','BiasLearnRateFactor',0)];

%design common path
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

%design the Critic Network
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

%plot figure
%figure
%plot(criticNetwork)

%set the Critic Network options
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1,'UseDevice','gpu');

% set critic values
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'action'},criticOpts);

%design Actor Network
actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','observation')
    fullyConnectedLayer(400,'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(300,'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(1,'Name','ActorFC3')
    tanhLayer('Name','ActorTanh')
    scalingLayer('Name','ActorScaling','Scale',max(actInfo.UpperLimit))];

%set the Actor Network options
actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1,'UseDevice','gpu');

% set actor values
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'ActorScaling'},actorOptions);

%set the DDPG options
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',128);


%generate DDPG agent
agent = rlDDPGAgent(actor,critic,agentOpts);

%set max episote and step size
maxepisodes = 5000;
maxsteps = ceil(Tf/Ts);

%set train options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'ScoreAveragingWindowLength',5,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeReward',...
    'StopTrainingValue',-0.5,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',-2500);

doTraining = true; 

if doTraining
    % Train the DDPG agent.
    trainingStats = train(agent,env,trainOpts);
    save 'v_5_IHA_NEW_Plant_DDPG.mat' agent;
else
    % Load the pretrained agent for the example.
    load('v_5_IHA_NEW_Plant_DDPG.mat','agent')
end


function in = localResetFcn(in)
%reset the integrator for X
blk = 'v_5_IHA_NEW_Plant/Plant(DRD-İHA)/Actual_X';
in = setBlockParameter(in,blk,'InitialCondition',num2str(0));

%reset the integrator for Y
blk = 'v_5_IHA_NEW_Plant/Plant(DRD-İHA)/Actual_Y';
in = setBlockParameter(in,blk,'InitialCondition',num2str(0));

%reset the integrator for Z
blk = 'v_5_IHA_NEW_Plant/Plant(DRD-İHA)/Actual_Z';
in = setBlockParameter(in,blk,'InitialCondition',num2str(0));

%reset the integrator for Phi
blk = 'v_5_IHA_NEW_Plant/Plant(DRD-İHA)/Actual_Phi';
in = setBlockParameter(in,blk,'InitialCondition',num2str(0));

%reset the integrator for Theta
blk = 'v_5_IHA_NEW_Plant/Plant(DRD-İHA)/Actual_Theta';
in = setBlockParameter(in,blk,'InitialCondition',num2str(0));

%reset the integrator for Psi
blk = 'v_5_IHA_NEW_Plant/Plant(DRD-İHA)/Actual_Psi';
in = setBlockParameter(in,blk,'InitialCondition',num2str(0));
end
