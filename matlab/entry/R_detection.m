function [R_index,R_value,NN50]=R_detection(y0,y_SM2,fs)	
    %R hullamok csucsai
	R_index=[];R_value=[];
	
	%Variabilitási indexek
	NN50=0;
	QS5=0;
	ind_QS=0;
    
	ind=0;
	max_RR_d=0.8*fs;
	delay=67;
    
	for i=round(max_RR_d/2)+1:round(length(y_SM2)-max_RR_d/2)-1
        if y_SM2(i)== max(y_SM2(i-round(max_RR_d/2):i+round(max_RR_d/2)))
            R_index=[R_index,i-delay];
            R_value=[R_value,y0(i-delay)];
            ind_R=length(R_index);
            if ind_R>2
                if abs(abs(R_index(ind_R)-R_index(ind_R-1))-abs(R_index(ind_R-1)-R_index(ind_R-2)))>0.05
                    NN50=NN50+1;
                end
            end
        end
    end                     
end