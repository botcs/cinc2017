folder_validation='..\..\..\af_challenge_2017\validation\';
folder_training='..\..\..\..\Challenge_2017\training2017\';

records='REFERENCE.csv';
filename=[folder_validation,records];
[numData,textData,rawData] = xlsread(filename);

yfit=[];
for i=1:length(textData)
    vr=challenge(char(strcat(folder_training,cellstr(textData(i,1)))));
    yfit=[yfit,vr];
end
        
%%
output_filename='answers.txt';
fid = fopen(output_filename,'wt');
for i=1:length(textData)
   fprintf(fid,'%s,%s\n',cell2mat(textData(i)),yfit(i)); 
end
fclose(fid);
%%
score_dataset(output_filename,folder_validation);