clc;
clear all;
close all;

folder='..\..\af_challenge_2017\validation\';
records='REFERENCE.csv';
filename=[folder,records];

[numData,textData,rawData] = xlsread(filename);

time_domain_features=[];
for i=1:length(textData)
    record_name=strjoin(strcat(folder,textData(i),'.mat'));
    data=load(record_name);
    y0=data.val;
    [pNN50,SDNN,RMSSD,...
     SDPR,RMSSD_PR,...
     SDQT,RMSSD_QT,...
     BPM,...
     pPR,pQR,pSR,pTR]=Time_Domain_Features(y0);
     
    time_domain_features=...
    [time_domain_features;...
     pNN50,SDNN,RMSSD,...
     BPM,pQR,pSR];
end

output_filename='time_domain_featurtes_v1.mat';
save(output_filename,'time_domain_features');
