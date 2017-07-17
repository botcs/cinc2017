% clc;
% clear all;
% close all;

train=1;

if train==1
    folder='..\..\..\Challenge_2017\training2017\';
    numrec=400;
elseif train ==0
    folder='..\..\af_challenge_2017\validation\';
    numrec=300;
end

records='REFERENCE.csv';
filename=[folder,records];

[numData,textData,rawData] = xlsread(filename);

features=[];

s=rng('shuffle');
r = randi(length(textData),1,600);
for ii=1:length(textData)
    %%
    record_name=strjoin(strcat(folder,textData(ii,1),'.mat'));
    data=load(record_name);
    y0=data.val;
    fs=300;
    %%
    N=2^(nextpow2(length(y0))-1);
    y0=y0(1:N);
    L  = length(y0); % signal length
    t  = (0:L-1)/fs;
    inputSignal(:,1,1) = y0';
    %%
    folderName = '../data/';
    tag = 'test/';
    
    % Import the data
    X = inputSignal;
    signalRange = [1 L]; % full range
    importData(X,folderName,tag,signalRange,fs);

    % perform Gabor decomposition
    Numb_points = L; % length of the signal
    Max_iterations = 100; % number of iterations
    runGabor(folderName,tag,Numb_points, Max_iterations);

    %%%%%%%%%%%%%%%%%%%%%%%%% Retrieve and display %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Retrieve information

    trialNum=1; % plot Trial number 

    gaborInfo{1} = getGaborData(folderName,tag,1);
    octave=gaborInfo{1,1}{1,1}.gaborData(1,:);
    modulus=gaborInfo{1,1}{1,1}.gaborData(4,:);
    
    frequency=gaborInfo{1,1}{1,1}.gaborData(2,:)/(L)*fs;
    gaborInfo{1,1}{1,1}.gaborData(6,:)=frequency;
    time=gaborInfo{1,1}{1,1}.gaborData(3,:)/fs;
    gaborInfo{1,1}{1,1}.gaborData(7,:)=time;
    phase=gaborInfo{1,1}{1,1}.gaborData(5,:)/pi*180;
    gaborInfo{1,1}{1,1}.gaborData(8,:)=phase;
    data=gaborInfo{1,1}{1,1}.gaborData(:,:)';

%     %% Reconstruct signal
%     wrap=0;
%     atomList=[]; % all atoms
% 
%     if isempty(atomList)
%         disp(['Reconstructing trial, all atoms']);
%     else
%         disp(['Reconstructing trial, atoms ' num2str(atomList(1)) ':' num2str(atomList(end))]);
%     end
% 
%     rSignal1 = reconstructSignalFromAtomsMPP(gaborInfo{1}{trialNum}.gaborData,L,wrap,atomList);
% 
%     % reconstruct energy
%     rEnergy1 = reconstructEnergyFromAtomsMPP(gaborInfo{1}{trialNum}.gaborData,L,wrap,atomList);
%     f = 0:fs/L:fs/2;
%     
%     % Plot reconstruction
%     ax(1)=subplot(411);
%     plot(t,inputSignal(:,trialNum,1)); 
%     title('Condition 1'); xlabel('Time (s)');
%     axis tight
% 
%     ax(2)=subplot(412);
%     plot(t,rSignal1,'k');
%     title('Reconstruction'); xlabel('Time (s)');
%     axis tight
% 
%     subplot(212);
%     pcolor(t,f,rEnergy1); shading interp;
%     xlabel('Time (s)'); ylabel('Frequency (Hz)');
%     axis tight
%     linkaxes(ax,'x');
    
    %%
    features=...
    [features;...
     textData(ii,1),textData(ii,2),...
     octave,modulus,frequency,time,phase];
 
    delete ([folderName,tag,'GaborMP/book.hdr']); 
    delete ([folderName,tag,'GaborMP/book.lst']);  
    delete ([folderName,tag,'GaborMP/mp0.bok.000']);  
 
    delete ([folderName,tag,'ImportData_SIG/GaborMP/local.ctl']);
    delete ([folderName,tag,'ImportData_SIG/sig.data.000']);
    delete ([folderName,tag,'ImportData_SIG/sig.hdr']);
    delete ([folderName,tag,'ImportData_SIG/sig.lst']);
    
    clearvars inputSignal;
    clearvars gaborInfo;

    ii
end
%%
if train==1
    training_features_v08=features;
    output_file='training_features_v08.mat';
    output_filename='training_features_v08';
elseif train ==0
    validation_features_v08=features;
    output_file='validation_features_v08.mat';
    output_filename='validation_features_v08';
end

save(['features/',output_file],output_filename);
