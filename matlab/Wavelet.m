
[phi1,psi1,phi2, psi2, XVAL] = wavefun('bior5.5',5); 
subplot(2,2,1) %pick the upper left of 4 subplots 
plot(phi1); 
subplot(2,2,2) %pick the upper right of 4 subplots 
plot(psi1); 
subplot(2,2,3) %pick the upper left of 4 subplots 
plot(phi2); 
subplot(2,2,4) %pick the upper right of 4 subplots 
plot(psi2);

[LoD,HiD,LoR,HiR] = wfilters('bior5.5');
subplot(221);
stem(LoD);
title('Lowpass Analysis Filter');
subplot(222);
stem(HiD);
title('Highpass Analysis Filter');
subplot(223);
stem(LoR);
title('Lowpass Synthesis Filter');
subplot(224);
stem(HiR);
title('Highpass Synthesis Filter');

% Alpha=1;
% w = gausswin(N,Alpha);

sigma = 5;
sz = 30;    % length of gaussFilter vector
x = linspace(-sz / 2, sz / 2, sz);
gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
gaussFilter = gaussFilter / sum (gaussFilter); % normalize

N=3;
[C,L] = wavedec2(y0,N,Lo_D,Hi_D)
[c,s]=wavedec2(y0,3,'bior5.5');

%% getDWT
[cD cA] = getDWT(y0,3,'bior5.5');
A5=cA(5,:);
yp = filter (gaussFilter,1, A5);