function [pNN50,SDNN,RMSSD,...
          SDPR,RMSSD_PR,...
          SDQT,RMSSD_RT,...
          BPM,...
          pPR,pQR,pSR,pTR]=Time_Domain_Features(y0)
    fs=300;
    
    %Pan Tompkins algorithm
    [y_LP,y_HP,y_DEV,y_SM2]=Pan_Tompkins(y0);
    
    %R detection
	[R_index,R_value,NN50]=R_detection(y0,y_SM2,fs);
    
    %BPM
    BPM=length(R_index)*60/(length(y0)/fs);
    
    %Q detection
	[Q_index,Q_value]=Q_detection(y0,y_HP,R_index,fs);
    
    %S detection
	[S_index,S_value]=S_detection(y0,y_HP,R_index,fs);

    %Delet QRS
%     y_PT_wave=QRS_delet(y0,Q_index,R_index,S_index,fs);
    
    %y_PT_wave smoothing
%     y_PT_wave_SM=PT_wave_smooth(y_PT_wave);
    
    %P detection
    [P_index,P_value]=P_detection(y0,R_index,BPM,fs);
    
    %T detection
    [T_index,T_value]=T_detection(y0,R_index,BPM,fs);

    %Plot ECG waves
    t=0:1/fs:(length(y0)-1)/fs;
%     plot(t,y0,...
%      t(P_index),P_value,'g*',...
%      t(R_index),R_value,'r*',...
%      t(T_index),T_value,'o');
 
%     plot(t,y0,...
%          t(P_index),P_value,'g*',...
%          t(Q_index),Q_value,'k*',...
%          t(R_index),R_value,'r*',...
%          t(S_index),S_value,'m*',...
%          t(T_index),T_value,'o');
    
    %% Features
    % ratio between NN50 and the total number of NN (normal-to-normal R-R interval) intervals
    pNN50=round(NN50/(length(R_index-2)),4);
    
    % standard deviation of normal R-R intervals
    % root mean square of successive R-R interval differences
    [SDNN,RMSSD]=TDA_NN(R_index);
    
    % standard deviation of normal P-R intervals
    % root mean square of successive P-R interval differences
    SDPR=[];
    RMSSD_PR=[];
    [SDPR,RMSSD_PR]=TDA_PR(R_index,P_index,fs);
    
    % standard deviation of normal R-T intervals !!!!! Q-T intervals
    % root mean square of successive R-T interval differences
    SDQT=[];
    RMSSD_RT=[];
    [SDQT,RMSSD_RT]=TDA_QT(R_index,T_index,fs);
    
    %Wave detection ratio
    pPR=length(P_index)/length(R_index);
    pQR=length(Q_index)/length(R_index);
    pSR=length(S_index)/length(R_index);
    pTR=length(T_index)/length(R_index);
    
end