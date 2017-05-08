function RMSSD=RMSSD_function(R_index)
    
% The expected value of NN interval
    NN=R_index(2:end)-R_index(1:end-1);
    N=length(NN);
    I=sqrt(1/N*sum(NN));
    

    
end