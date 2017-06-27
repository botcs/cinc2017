function [P_index,P_value]=P_detection(y0,R_index,BPM,fs)

c=60/BPM;% weighted heart rate(HR)

w=0;%0.1*c*fs; %QRS duration
v=0.08*c*fs;%ST duration
u=0.2*c*fs; %PR duration
z=0.4*c*fs; %QT duration
s=0.1*c*fs; %P wave duration

P_index=[];
P_value=[];
P_windows=round([R_index-0.5*w-u;R_index-0.5*w-u+s]);
P_windows(P_windows<=0)=1;
P1=P_windows(1,:);
P2=P_windows(2,:);

    for i=1:length(P_windows)
      yp=y0(P_windows(1,i):P_windows(2,i));

      len_yp=length(yp);
      % Simitas egy mozgo atlag szurovel
      win_yp=round(len_yp/20);
      delay=round(length(win_yp)/2);
      b_SM = 1/win_yp*ones(1,win_yp);
      a_SM = [1 zeros(1,win_yp-1)];

      yp_SM=filter(b_SM, a_SM, yp);
      mean_yp=mean(yp_SM)*ones(len_yp);

      tp=0:1/fs:(length(yp_SM)-1)/fs;
      pks1=[];locs1=[];pks2=[];locs2=[];

      [pks1,locs1] = findpeaks(yp_SM);
      [pks2,locs2] = findpeaks(-yp_SM);

      locs_yp=[locs1,locs2];
      pks_yp=[pks1,-pks2];

%       abs_pks_yp=abs((pks_yp+abs(min(yp_SM)))-(mean_yp(1)+abs(min(yp_SM))));
%       [M,I]=max(abs_pks_yp);

      sd_pks_yp=abs(locs_yp-len_yp/2);
      [M,I]=min(sd_pks_yp);

      f_locs_yp=locs_yp(I);
      f_pks_yp=pks_yp(I);
      
      P_index=[P_index,R_index(i)-2*len_yp+f_locs_yp];
      P_value=[P_value,f_pks_yp];
      
%       t=0:1/fs:(length(y0)-1)/fs;
%       plot(t,y0,t(P_index),P_value,'r*',...
%            t(P1),200,'g*',...
%            t(P2),200,'k*');

%       figure(2); 
%       plot(tp,yp,'k-',tp,yp_SM,'b-',...
%           tp(locs1),pks1,'m*',...
%           tp(locs2),-pks2,'g*',...
%           tp(f_locs_yp),f_pks_yp,'ro',...
%           tp,mean_yp,'r-')

    end

end

% function [P_index,P_value]=P_detection(y0,R_index,y_PT_wave_SM,fs)
%     
%     %P hullam csucsai
%     P_index=[];P_value=[];
% 
% 	max_RR_d=max(R_index(2:end)-R_index(1:end-1));
%     max_PR_d=round((length(y0)/length(R_index))/4);
%     y_P_wave_1=y_PT_wave_SM.*y_PT_wave_SM;
%     delay=8;
%     
%     j=1;
%     for i=round(max_RR_d/2)+1:round(length(y_P_wave_1)-max_RR_d/2)-1
%         start=R_index(j)-max_PR_d;
%         stop=R_index(j);
%         if i>=start && i<=stop && y_P_wave_1(i)==max(y_P_wave_1(start:stop))
%             P_index=[P_index i-delay];
%             P_value=[P_value y0(i-delay)];
%         elseif i>stop
%             j=j+1;
%         end
%     end
% end