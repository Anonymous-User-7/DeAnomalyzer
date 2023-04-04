clear
clc
%% Init
main_function()
%% Main Function
function main_function()
    T = readtable('GD_ReRun/LOF.csv', 'ReadVariableNames', false);
    filename = string(T.Var1);
    anomaly = T.Var2;
    [threshold,lof] = LOF_bin(filename,anomaly);
    
    fid = fopen('GD_ReRun/LOF_M_Thr.csv', 'w');
    fprintf(fid, '%f', threshold);
    fclose(fid);
    
    %Default
    outlier=lof>=2;
    writefilename = 'Labels/LOF_Matlab/' + filename+ '_Default.csv';
    csvwrite(writefilename,outlier')
    
    %Modified
    if threshold == 0
        threshold = 2;
    end
    outlier1=lof>=threshold;
    writefilename = 'Labels/LOF_Matlab/' + filename + '_Mod.csv';
    csvwrite(writefilename,outlier1')
end
 %% Read File
function [X, y] = csvfileread(readfilename)
    readfilename = readfilename+".csv";
    T = readtable(readfilename, 'ReadVariableNames', true);
    ColIndex = find(strcmp(T.Properties.VariableNames, 'target'), 1);
    A = table2array(T);
    A(any(isnan(A), 2), :) = [];
    target=A(:, ColIndex);
    A(:, ColIndex)=[];
    X = A;
    y = target;
end
function [X, y] = matfileread(readfilename)
    readfilename = readfilename+".mat";
    A = load(readfilename);
    X = A.X;
    y = A.y;
end
%% Search Threshold
function [mid, lof]  = LOF_bin(filename,anomaly)
    readfilename = sprintf('Dataset/%s',filename);
    mid = 0;
    try
        [X, y] = csvfileread(readfilename);
    catch
        [X, y] = matfileread(readfilename);
    end
    if size(X, 1) < size(X,2)*2
        disp("Dataset Dimention Error.\n")
        return
    end
    k = 20;
    [suspicious_index lof] = LOF(X, k);
    start = min(lof);
    last = max(lof);
    iteration = 1;
    while true 
        iteration = iteration+1;
        if iteration>10
            return
        end
        middle = (start+last)/2;
        pred_new = lof;
        for  j = 1:length(lof)
            if lof(j)> middle
                pred_new(j) = 1;
            else
                pred_new(j) = 0;
            end
        end
        anomaly_bin = (sum(pred_new == 1)/length(pred_new));
        if (abs(anomaly_bin - anomaly) < 0.000001)
            mid = middle;
        elseif anomaly_bin > anomaly
            start = middle;
        else
            last = middle;
        end
    end
end

