function y_PT_wave=QRS_delet(y0,Q_index,R_index,S_index,fs)
    
	max_RR_d=max(R_index(2:end)-R_index(1:end-1));
    max_QRS_d=round(max_RR_d/4);
    j=1;k=1;
    y_PT_wave=[];
    for i=round(max_RR_d/2)+1:round(length(y0)-max_RR_d/2)-1
        y_PT_wave(i)=y0(i);
        if k<=length(S_index) && j<=length(Q_index)
            if Q_index(j)<=i && i<=S_index(k) && S_index(k)-Q_index(j)<max_QRS_d
                y_PT_wave(i)=y0(Q_index(j))+...
                    (y0(S_index(k))-y0(Q_index(j)))/(S_index(k)-Q_index(j))*...
                    (i-Q_index(j));
            elseif Q_index(j)>S_index(k)
                k=k+1;
            elseif S_index(k)-Q_index(j)>=max_QRS_d
                j=j+1;
            elseif i>S_index(k)
                j=j+1;
                k=k+1;
            end
        end
    end

end