function [T_index,T_value]=T_detection(y0,R_index,BPM,fs)

c=60/BPM;% weighted heart rate(HR)

w=0.1*c*fs; %QRS duration
v=0.08*c*fs;%ST duration
u=0.2*c*fs; %PR duration
z=0.4*c*fs; %QT duration
s=0.1*c*fs; %P wave duration

T_index=[];
T_value=[];
T_windows=round([R_index+0.5*w+v;R_index-0.5*w+z]);
T_windows(T_windows<=0)=1;
T1=T_windows(1,:);
T2=T_windows(2,:);


    for i=1:length(T_windows)
      yt=y0(T_windows(1,i):T_windows(2,i));

      len_yt=length(yt);
      % Simitas egy mozgo atlag szurovel
      win_yt=round(len_yt/20);
      delay=round(length(win_yt)/2);
      b_SM = 1/win_yt*ones(1,win_yt);
      a_SM = [1 zeros(1,win_yt-1)];

      yt_SM=filter(b_SM, a_SM, yt);
      mean_yt=mean(yt_SM)*ones(len_yt);

      tp=0:1/fs:(length(yt_SM)-1)/fs;
      pks1=[];locs1=[];pks2=[];locs2=[];

      [pks1,locs1] = findpeaks(yt_SM);
      [pks2,locs2] = findpeaks(-yt_SM);

      locs_yt=[locs1,locs2];
      pks_yt=[pks1,-pks2];

%       abs_pks_yt=abs((pks_yt+abs(min(yt_SM)))-(mean_yt(1)+abs(min(yt_SM))));
%       [M,I]=max(abs_pks_yt);

      sd_pks_yt=abs(locs_yt-len_yt/2);
      [M,I]=min(sd_pks_yt);

      f_locs_yt=locs_yt(I);
      f_pks_yt=pks_yt(I);
      
      T_index=[T_index,R_index(i)+f_locs_yt+round(len_yt/5)];
      T_value=[T_value,f_pks_yt];
      
%       t=0:1/fs:(length(y0)-1)/fs;
%       plot(t,y0,t(T_index),T_value,'r*',...
%            t(T1),200,'g*',...
%            t(T2),200,'k*');

%       figure(2); 
%       plot(tp,yt,'k-',tp,yt_SM,'b-',...
%           tp(locs1),pks1,'m*',...
%           tp(locs2),-pks2,'g*',...
%           tp(f_locs_yt),f_pks_yt,'ro',...
%           tp,mean_yt,'r-')

    end

end

% function [T_index,T_value]=T_detection(y0,R_index,y_PT_wave_SM,fs)
%     
%     %T hullam csucsai
%     T_index=[];T_value=[];
% 
% 	max_RR_d=max(R_index(2:end)-R_index(1:end-1));
%     max_RT_d=round((length(y0)/length(R_index))/2);
%     delay=14;
%     
%     j=1;
%     for i=round(max_RR_d/2)+1:round(length(y_PT_wave_SM)-max_RR_d/2)-1
%         start=R_index(j);
%         stop=R_index(j)+max_RT_d;
%         if i>=start && i<=stop && y_PT_wave_SM(i)==max(y_PT_wave_SM(start:stop))
%             T_index=[T_index i-delay];
%             T_value=[T_value y0(i-delay)];
%         elseif i>stop
%             j=j+1;
%         end
%     end
% end