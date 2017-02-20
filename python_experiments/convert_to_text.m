files=dir('training2017');
for f=1:size(files,1)
    fname=files(f).name;
    Ext=strsplit(fname,'.');
    if size(Ext,2)==2
       if strcmp(Ext{2},'mat')==1
            load(['training2017/',fname]);
            fileID = fopen(['text/',Ext{1},'.txt'],'w');
            for v=1:size(val,2)
                fprintf(fileID,'%12.1f,',val(v));
            end
            fclose(fileID);
        end
    end
    
end
