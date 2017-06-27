% clc;
% clear all;
% close all;

folder='..\..\..\Challenge_2017\training2017\';
% folder='..\..\af_challenge_2017\validation\';
records='REFERENCE.csv';
filename=[folder,records];

[numData,textData,rawData] = xlsread(filename);

training_features_v01=[]
s=rng('shuffle');
r = randi(length(textData),1,600);
for ii=1:400%length(textData)%length(r)
    jj=ii;%r(ii);
    record_name=strjoin(strcat(folder,textData(jj,1),'.mat'));
    data=load(record_name);
    y0=data.val;
    
    fs=300;
    t=0:1/fs:(length(y0)-1)/fs;
%     plot(t,y0);
%     ylabel('Amplitude');
%     xlabel('Time (s)'); 
%     title([textData(jj,1),textData(jj,2)]);
    
    [pNN50,SDNN,RMSSD,...
     SDPR,RMSSD_PR,...
     SDQT,RMSSD_QT,...
     BPM,...
     pPR,pQR,pSR,pTR,...
     cD1, cD2, cD3, cA3,...
     selected_frequencies]=Time_Domain_Features(y0);
      
    training_features_v01=...
    [training_features_v01;...
     textData(jj,1),textData(jj,2),...
     pNN50,SDNN,RMSSD,...
     SDPR,RMSSD_PR,...
     SDQT,RMSSD_QT,...
     BPM,...
     pPR,pQR,pSR,pTR,...
     cD1, cD2, cD3, cA3,...
     selected_frequencies];
    
    ii
end
%% 
 output_filename='training_features_v01.mat';
 save(output_filename,'training_features_v01');
