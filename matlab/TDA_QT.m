function [SDQT,RMSSD_QT]=TDA_QT(Q_index,T_index,fs)
% The expected value of NN interval
    QT=[];
    max_RR_d=0.8*fs;
    j=1;
    for i=1:length(Q_index)
        diff_QT=Q_index(i)-T_index(j);
        if diff_QT>0 && diff_QT<max_RR_d
            QT=[QT diff_QT];
            j=j+1;
        elseif diff_QT>max_RR_d && (j+2)<length(T_index)
            j=j+2;
        end
    end
    
    N=length(QT);
    I=sqrt(1/N*sum(QT));
    
% Standard deviation of normal R-R intervals
    PR2=pow2((QT-I),2);
    SDQT=sqrt(1/N*sum(PR2));
    
% Root mean square of successive P-R interval differences
    PRe=pow2((QT(2:end)-QT(1:end-1)),2);
    RMSSD_QT=sqrt(1/(N-1)*sum(PRe));
end 