function     [mean_cD1, mean_cD2, mean_cD3, mean_cA3] = Wavelet_Features(y0, R_index)

m=([R_index 0]-[0 R_index])/2;
window=mean(m(2:length(R_index)));
mean_cD1=0;
mean_cD2=0;
mean_cD3=0;
mean_cA3=0;
    for i=1:length(R_index)
        if (window < R_index(i) && R_index(i)< length(y0)-window)
            input_w=y0(ceil(R_index(i)-window):ceil(R_index(i)+window));
            input_w=resample(input_w,100,length(input_w));
            [cA1,cD1] = dwt(input_w,'bior5.5');
            mean_cD1=mean_cD1+cD1/length(R_index);
            [cA2,cD2] = dwt(cA1,'bior5.5');
            mean_cD2=mean_cD2+cD2/length(R_index);
            [cA3,cD3] = dwt(cA2,'bior5.5');
             mean_cA3=mean_cA3+cA3/length(R_index);
             mean_cD3=mean_cD3+cD3/length(R_index);
        end
    end
end

