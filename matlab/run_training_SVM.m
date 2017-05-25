load('time_domain_features_v6.mat');
load('time_domain_features_validation.mat');

train_set=time_domain_features_v6;
len_train=length(train_set);

folder_validation='..\..\af_challenge_2017\validation';
folder_training='..\..\..\Challenge_2017\training2017';
records='REFERENCE.csv';
filename_training=[folder_training filesep records];
filename_validation=[folder_validation filesep records];

[numData_v,textData_v,rawData_v] = xlsread(filename_validation);
[numData_t,textData_t,rawData_t] = xlsread(filename_training);

Y0_val=time_domain_features_validation(:,2);
Y_val=time_domain_features_validation(:,2);
Y_val(cell2mat(cellfun(@(elem) elem == 'N', Y_val(:, :),'UniformOutput', false))) = {1};
Y_val(cell2mat(cellfun(@(elem) elem == 'A', Y_val(:, :),'UniformOutput', false))) = {2};
Y_val(cell2mat(cellfun(@(elem) elem == 'O', Y_val(:, :),'UniformOutput', false))) = {3};
Y_val(cell2mat(cellfun(@(elem) elem == '~', Y_val(:, :),'UniformOutput', false))) = {4};
Y_val=cell2mat(Y_val);
X_val=real(cell2mat(time_domain_features_validation(:,3:end)));

Y0=train_set(:,2);
Y=train_set(:,2);
Y(cell2mat(cellfun(@(elem) elem == 'N', Y(:, :),'UniformOutput', false))) = {1};
Y(cell2mat(cellfun(@(elem) elem == 'A', Y(:, :),'UniformOutput', false))) = {2};
Y(cell2mat(cellfun(@(elem) elem == 'O', Y(:, :),'UniformOutput', false))) = {3};
Y(cell2mat(cellfun(@(elem) elem == '~', Y(:, :),'UniformOutput', false))) = {4};
Y=cell2mat(Y);
X=real(cell2mat(train_set(:,3:end)));

%% Training
SVMModel = fitcecoc(X, Y)
CVSVMModel = crossval(SVMModel);
%%time_domain_featurtes
oosLoss = kfoldLoss(CVSVMModel);
FirstModel = CVSVMModel.Trained{1};
[yfit,score] = predict(FirstModel, X_val);
%%
yfit=num2str(yfit);
yfit(yfit=='1')='N';
yfit(yfit=='2')='A';
yfit(yfit=='3')='O';
yfit(yfit=='4')='~';
%%%
output_filename='answers.txt';
fid = fopen(output_filename,'wt');
for i=1:length(yfit)
   fprintf(fid,'%s,%s\n',cell2mat(rawData_v(i,1)),yfit(i)); 
end
fclose(fid);

score_dataset(output_filename,folder_validation);
%%
yfit_c=cellstr(yfit);
index=[];value_val=[];value_fit=[];
for i=1:length(yfit)
    if ~isequal(Y0_val(i),yfit_c(i))
        index=[index,i];
        value_val=[value_val,Y0_val(i)];
        value_fit=[value_fit,yfit_c(i)];
    end
end
%%

value_val_mat=cell2mat(value_val');
value_fit_mat=cell2mat(value_fit');
index_mat=num2str(index');
eval_matrix=[strcat(index_mat,',',value_val_mat,',',value_fit_mat)];

 output_filename='CVSVMModel.mat';
 save(output_filename,'CVSVMModel');
