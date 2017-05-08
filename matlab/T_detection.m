function [T_index,T_value]=T_detection(y0,R_index,y_PT_wave_SM,fs)
    
    %T hullam csucsai
    T_index=[];T_value=[];

	max_RR_d=max(R_index(2:end)-R_index(1:end-1));
    max_RT_d=round((length(y0)/length(R_index))/2);
    delay=14;
    
    j=1;
    for i=round(max_RR_d/2)+1:round(length(y_PT_wave_SM)-max_RR_d/2)-1
        start=R_index(j);
        stop=R_index(j)+max_RT_d;
        if i>=start && i<=stop && y_PT_wave_SM(i)==max(y_PT_wave_SM(start:stop))
            T_index=[T_index i-delay];
            T_value=[T_value y0(i-delay)];
        elseif i>stop
            j=j+1;
        end
    end
end