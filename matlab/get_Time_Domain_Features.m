% clc;
% clear all;
% close all;

train=0;

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
for ii=1:numrec%length(textData)%length(r)
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
      
    features=...
    [features;...
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
if train==1
    training_features_v04=features;
    output_file='training_features_v04.mat';
    output_filename='training_features_v04';
elseif train ==0
    validation_features_v04=features;
    output_file='validation_features_v04.mat';
    output_filename='validation_features_v04';
end

save(output_file,output_filename);

%%

mean_ecg_fft=cell2mat(features(:,19:19));

w=size(mean_ecg_fft,2);
mean_ecg_fft_n=zeros(1,w);
mean_ecg_fft_a=zeros(1,w);
mean_ecg_fft_o=zeros(1,w);
mean_ecg_fft_s=zeros(1,w);

index_n=0;
index_a=0;
index_o=0;
index_s=0;

max_f=30;
t=0:(max_f)/89:max_f;

for i=1:numrec
    if strcmp(features(i,2),'N')==1;
        color='r';
        index_n=index_n+1;
        mean_ecg_fft_n=(mean_ecg_fft_n+mean_ecg_fft(i,:));
    elseif strcmp(features(i,2),'A')==1;
        color='b';
        index_a=index_a+1;
        mean_ecg_fft_a=(mean_ecg_fft_a+mean_ecg_fft(i,:));
    elseif strcmp(features(i,2),'O')==1;
        color='g';
        index_o=index_o+1;
        mean_ecg_fft_o=(mean_ecg_fft_o+mean_ecg_fft(i,:));
    elseif strcmp(features(i,2),'~')==1;
        color='m';
        index_s=index_s+1;
        mean_ecg_fft_s=(mean_ecg_fft_s+mean_ecg_fft(i,:));
    end
    figure(2)
    plot(t,mean_ecg_fft(i,:),color);
    hold on
end

mean_ecg_fft_n=mean_ecg_fft_n/index_n;
mean_ecg_fft_a=mean_ecg_fft_a/index_a;
mean_ecg_fft_o=mean_ecg_fft_o/index_o;
mean_ecg_fft_s=mean_ecg_fft_s/index_s;

figure(3)
plot(t,mean_ecg_fft_n,'r',...
     t,mean_ecg_fft_a,'b',...
     t,mean_ecg_fft_o,'g',...
     t,mean_ecg_fft_s,'m');
 
legend('Normal','AF','Others','Noisy');
ylabel('Amplitude');
xlabel('Frequency of mean RR intervalls (Hz)'); 