folder='..\..\..\Challenge_2017\training2017\';
output_file='training_features_v05.mat';
features_data=importdata(['features/',output_file]);

for i=1:8528
    num=int2str(i);
    if(i<10)
        num=strcat('A0000',num);
    elseif i<100
        num=strcat('A000',num);
    elseif i<1000
        num=strcat('A00',num);
    else
        num=strcat('A0',num);
    end
     
    record_name=([folder,num,'.mat'])
    load(record_name)
    features=features_data(i,:);
    save(record_name,'val');
    save(record_name,'features','-append');
    i
end