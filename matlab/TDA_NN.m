function [SDNN,RMSSD]=TDA_NN(R_index)
    
% The expected value of NN interval
    NN=R_index(2:end)-R_index(1:end-1);
    N=length(NN);
    I=sqrt(1/N*sum(NN));
    
% Standard deviation of normal R-R intervals
    NN2=pow2((NN-I),2);
    SDNN=sqrt(1/N*sum(NN2));

% Root mean square of successive R-R interval differences
    NNe=pow2((NN(2:end)-NN(1:end-1)),2);
    RMSSD=sqrt(1/(N-1)*sum(NNe));
    
end