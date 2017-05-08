function [P_index,P_value]=P_detection(y0,R_index,y_PT_wave_SM,fs)
    
    %P hullam csucsai
    P_index=[];P_value=[];

	max_RR_d=max(R_index(2:end)-R_index(1:end-1));
    max_PR_d=round((length(y0)/length(R_index))/4);
    delay=8;
    
    j=1;
    for i=round(max_RR_d/2)+1:round(length(y_PT_wave_SM)-max_RR_d/2)-1
        start=R_index(j)-max_PR_d;
        stop=R_index(j);
        if i>=start && i<=stop && y_PT_wave_SM(i)==max(y_PT_wave_SM(start:stop))
            P_index=[P_index i-delay];
            P_value=[P_value y0(i-delay)];
        elseif i>stop
            j=j+1;
        end
    end
end