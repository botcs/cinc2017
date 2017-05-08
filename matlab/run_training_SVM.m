load('time_domain_features_v1.mat');

folder='..\..\af_challenge_2017\validation';
records='REFERENCE.csv';
filename=[folder filesep records];

[numData,textData,rawData] = xlsread(filename);
Y0=rawData(:,2);
Y=rawData(:,2)';
Y(cell2mat(cellfun(@(elem) elem == 'N', Y(:, :),'UniformOutput', false))) = {1};
Y(cell2mat(cellfun(@(elem) elem == 'A', Y(:, :),'UniformOutput', false))) = {2};
Y(cell2mat(cellfun(@(elem) elem == 'O', Y(:, :),'UniformOutput', false))) = {3};
Y(cell2mat(cellfun(@(elem) elem == '~', Y(:, :),'UniformOutput', false))) = {4};
Y=cell2mat(Y);
X=real(time_domain_features);

%% Training
SVMModel = fitcecoc(X, Y)
CVSVMModel = crossval(SVMModel);

oosLoss = kfoldLoss(CVSVMModel);
FirstModel = CVSVMModel.Trained{1};
[yfit,score] = predict(FirstModel, X);

output=strrep(output,'1','N');
output=strrep(output,'2','A');
output=strrep(output,'3','O');
output=strrep(output,'4','~');
output=strrep(output,'  ','');

output_filename='yfit.txt';
dlmwrite(output_filename,output);

score_dataset(output_filename,folder);
