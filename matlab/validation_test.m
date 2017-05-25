
folder_validation='..\..\..\af_challenge_2017\validation\';
folder_training='..\..\..\..\Challenge_2017\training2017\';

records='REFERENCE.csv';
filename=[folder_validation,records];
[numData,textData,rawData] = xlsread(filename);

yfit=[];
for i=1:length(textData)
    vr=challenge(strcat(folder_training,cellstr(textData(i,1))));
    yfit=[yfit,vr];
end
        
yfit=num2str(yfit);
yfit(yfit=='1')='N';
yfit(yfit=='2')='A';
yfit(yfit=='3')='O';
yfit(yfit=='4')='~';
%%%
output_filename='output.txt';
fid = fopen(output_filename,'wt');
for i=1:length(textData)
   fprintf(fid,'%s,%s\n',cell2mat(textData(i)),yfit(i)); 
end
fclose(fid);
