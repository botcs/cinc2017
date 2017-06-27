function [selected_frequencies]=Frequency_Domain_Features(y0, R_index, fs)

f1=17;
f2=27;
f3=33;
f4=38;
f5=134;

mean_f1=0;
mean_f2=0;
mean_f3=0;
mean_f4=0;
mean_f5=0;

m=([R_index 0]-[0 R_index])/2;
window=mean(m(2:length(R_index)));


for i=1:length(R_index)
    if (window < R_index(i) && R_index(i)< length(y0)-window)
        input_w=y0(ceil(R_index(i)-window):ceil(R_index(i)+window));
   
        Ny = length(input_w);			
        NFFT = 2^nextpow2(Ny);
        Y = fft(input_w,NFFT);
        f = fs/2*linspace(0,1,NFFT/2+1);
        X=f(1:end);
        Y=abs(Y(1:(NFFT/2+1)));
        
        mean_f1=mean_f1+Y(round(f1/max(X)*(NFFT)/2))/length(R_index);
        mean_f2=mean_f2+Y(round(f2/max(X)*(NFFT)/2))/length(R_index);
        mean_f3=mean_f3+Y(round(f3/max(X)*(NFFT)/2))/length(R_index);
        mean_f4=mean_f4+Y(round(f4/max(X)*(NFFT)/2))/length(R_index);
        mean_f5=mean_f5+Y(round(f5/max(X)*(NFFT)/2))/length(R_index);      
    end
end

selected_frequencies=[mean_f1,mean_f2,mean_f3,mean_f4,mean_f5];
end