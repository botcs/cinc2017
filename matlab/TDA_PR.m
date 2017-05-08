function [SDPR,RMSSD_PR]=TDA_PR(R_index,P_index,fs)
% The expected value of NN interval
    PR=[];
    max_RR_d=0.8*fs;
    j=1;
    for i=1:length(R_index)
        diff_PR=R_index(i)-P_index(j);
        if diff_PR>0 && diff_PR<max_RR_d
            PR=[PR diff_PR];
            j=j+1;
        elseif diff_PR>max_RR_d && (j+2)<length(P_index)
            j=j+2;
        end
    end
    
    N=length(PR);
    I=sqrt(1/N*sum(PR));
    
% Standard deviation of normal R-R intervals
    PR2=pow2((PR-I),2);
    SDPR=sqrt(1/N*sum(PR2));
    
% Root mean square of successive P-R interval differences
    PRe=pow2((PR(2:end)-PR(1:end-1)),2);
    RMSSD_PR=sqrt(1/(N-1)*sum(PRe));
end 