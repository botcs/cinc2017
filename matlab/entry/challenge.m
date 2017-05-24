function classifyResult = challenge(recordName)
    
    load CVSVMModel;
    records='REFERENCE.csv';
    
    [tm,ecg,fs,siginfo]=rdmat(strcat(recordName);
    
    [pNN50,SDNN,RMSSD,...
     SDPR,RMSSD_PR,...
     SDQT,RMSSD_QT,...
     BPM,...
     pPR,pQR,pSR,pTR]=Time_Domain_Features(ecg);
 
    time_domain_features=...
    abs([pNN50,SDNN,RMSSD,...
     SDPR,RMSSD_PR,...
     SDQT,RMSSD_QT,...
     BPM,...
     pPR,pQR,pSR,pTR]);
 
    feature_vector=[time_domain_features];
    FirstModel = CVSVMModel.Trained{1};
    [classifyResult,~] = predict(FirstModel, feature_vector)
end