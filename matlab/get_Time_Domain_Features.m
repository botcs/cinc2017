% clc;
% clear all;
% close all;

folder='..\..\..\Challenge_2017\training2017\';
% folder='..\..\af_challenge_2017\validation\';
records='REFERENCE.csv';
filename=[folder,records];

[numData,textData,rawData] = xlsread(filename);

time_domain_features_v4=[];
rng('shuffle')
r = randi([1 length(textData)],1,600);
for ii=1:length(textData)
    record_name=strjoin(strcat(folder,textData(ii),'.mat'));
    data=load(record_name);
    y0=data.val;
    fs=300;
    t=0:1/fs:(length(y0)-1)/fs;
    plot(t,y0);
    ylabel('Amplitude');
    xlabel('Time (s)'); 
    
    [pNN50,SDNN,RMSSD,...
     SDPR,RMSSD_PR,...
     SDQT,RMSSD_QT,...
     BPM,...
     pPR,pQR,pSR,pTR]=Time_Domain_Features(y0);
     
    time_domain_features_v4=...
    [time_domain_features_v4;...
     pNN50,SDNN,RMSSD,...
     SDPR,RMSSD_PR,...
     SDQT,RMSSD_QT,...
     BPM,...
     pPR,pQR,pSR,pTR];
    
    ii
end
%% 
 output_filename='time_domain_features_v4.mat';
 save(output_filename,'time_domain_features_v4');
