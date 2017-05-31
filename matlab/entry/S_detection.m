function [S_index,S_value]=S_detection(y0,y_HP,R_index,fs)	
	%S hullamok csucsai
    S_index=[];S_value=[];
	
	%Variabilitási indexek
	QS5=0;
	ind_QS=0;
	ind=0;
    
	max_RR_d=max(R_index(2:end)-R_index(1:end-1));
    max_QRS_d=round(max_RR_d/4);
    j=1;
    
    y_HP3=y_HP.*y_HP.*y_HP;
    delay=34;
    R_index2=R_index+delay;
    
    for i=round(max_RR_d/2)+1:round(length(y_HP3)-max_RR_d/2)-1
        start=R_index2(j);
        stop=R_index2(j)+max_QRS_d/2;
        
        if start<=i && i<=stop && y_HP3(i)== min(y_HP3(ceil(start):ceil(stop)))
            S_index=[S_index i-delay];
            S_value=[S_value y0(i-delay)];    
            
            if j<length(R_index)
                j=j+1;
            end
        end
    end
end