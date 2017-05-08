function [y_LP,y_HP,y_DEV,y_SM2]=Pan_Tompkins(input)
    
    %%Szuro Koefficiensek
    % Alul atereszto szuro definialasa
    b_LPF=[1/72 zeros(1,8) -1/36 zeros(1,8) 1/72];
    a_LPF=[1 -2 1 zeros(1,16)];
    % Felul atereszto szuro definialasa
    b_HPF=[-1/49 zeros(1,24) 1 -1 zeros(1,22) 1/49];
    a_HPF=[1 -1 zeros(1,48)];
    % Differenciallas
    b_DEV=[1/3.6 0 1/8 0 -1/8 0 -1/3.6];
    a_DEV=[1 zeros(1,6)];
    % Simitas egy mozgo atlag szurovel
    b_SM = 1/30*ones(1,30);
    a_SM = [1 zeros(1,29)];

    %% Pan Tompkins algoritmus
	%LP filter
	y_LP = filter(b_LPF, a_LPF, input);

	% HP Filter
	y_HP =filter(b_HPF, a_HPF, y_LP);

	% Differentiation
	y_DEV =filter(b_DEV, a_DEV, y_HP);

	% Squaring
	y_SQ = y_DEV.*y_DEV;

	% Smoothing
	y_SM1 = filter(b_SM, a_SM, y_SQ);
	y_SM2 = filter(b_SM, a_SM, y_SM1);
end