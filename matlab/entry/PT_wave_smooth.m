function y_PT_wave_SM=PT_wave_smooth(y_PT_wave)
    % Simitas egy mozgo atlag szurovel
    b_SM = 1/30*ones(1,30);
    a_SM = [1 zeros(1,29)];
    
    % Smoothing
	y_PT_wave_SM = filter(b_SM, a_SM, y_PT_wave);
end
