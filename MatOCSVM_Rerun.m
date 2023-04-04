clear
clc
%% Init
main_function()
%% Main Function
function main_function()
    filename = 'GD_ReRun/MatOCSVM.csv';
    opts = detectImportOptions(filename);
    opts = setvartype(opts,'char');  % or 'string'
    T = readtable(filename,opts);
    T = table2array(T);
    for i = 1:size(T,1)
%         fprintf("%d\n", i)
        parameters = [];
        ContaminationFraction.name = "ContaminationFraction";
        ContaminationFraction.default = cell2mat(T(i,2));
        KernelScale.name = "KernelScale";
        KernelScale.default = cell2mat(T(i,3));
        Lambda.name = "Lambda";
        Lambda.default = cell2mat(T(i,4));
        NumExpansionDimensions.name = "NumExpansionDimensions";
        NumExpansionDimensions.default = cell2mat(T(i,5));
        StandardizeData.name = "StandardizeData";
        StandardizeData.default = str2double(cell2mat(T(i,6)));
        BetaTolerance.name = "BetaTolerance";
        BetaTolerance.default = str2double(cell2mat(T(i,7)));
        GradientTolerance.name = "GradientTolerance";
        GradientTolerance.default = str2double(cell2mat(T(i,8)));
        IterationLimit.name = "IterationLimit";
        IterationLimit.default = str2double(cell2mat(T(i,9)));
        
        parameters = [ContaminationFraction, KernelScale, Lambda, NumExpansionDimensions, StandardizeData, BetaTolerance, GradientTolerance, IterationLimit];

        OCSVM(cell2mat(T(i,1)), parameters);
    end
end

%% Read File
function [X, y] = csvfileread(readfilename)
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
    A = load(readfilename);
    X = A.X;
    y = A.y;
end

%% OCSVM
function OCSVM(filename, parameters)
    readfilename = sprintf('Dataset/%s', filename);
    
    if isfile(sprintf('Dataset/%s.csv', filename))
        [X, y] = csvfileread(sprintf('Dataset/%s.csv', filename));
    end
    if isfile(sprintf('Dataset/%s.mat', filename))
        [X, y] = matfileread(sprintf('Dataset/%s.mat', filename));
    end
    
    runOCSVM(filename, X, y, parameters);
            
  end
%% Run OCSVM
function runOCSVM(filename_with_extension, X, y, params)
    filename_char = convertStringsToChars(filename_with_extension);
    filename = filename_char;
    labelFile = "Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+filename + "_" + params(1).default + "_" + params(2).default + "_" + params(3).default + "_" + params(4).default + "_" + params(5).default + "_" + params(6).default + "_" + params(7).default + "_" + params(8).default + ".csv";
    if isfile(labelFile)
       return
    end
%     labelFile
    p1 = params(1).default;
    p2 = params(2).default;
    p3 = params(3).default;
    p4 = params(4).default;
    p5 = params(5).default;
    p6 = params(6).default;
    p7 = params(7).default;
    p8 = params(8).default;
    if string(p1) == "LOF"
        percentage_table = readtable("Stats/SkPercentage.csv");
        percentage_table_file = percentage_table(string(percentage_table.Filename)==filename, :);
        p1 = percentage_table_file.LOF;
    elseif string(p1) == "IF"
        percentage_table = readtable("Stats/SkPercentage.csv");
        percentage_table_file = percentage_table(string(percentage_table.Filename)==filename, :);
        p1 = percentage_table_file.IF;        
    else
        p1 = str2double(p1);
    end
    if string(p2) ~= "auto"
        p2 = str2double(p2);
    end
    if string(p3) ~= "auto"
        p3 = str2double(p3);
    end
    if string(p4) ~= "auto"
        p4 = str2double(p4);
    end
    sX = string([1:size(X, 2)]);
    outliersSet = [];
    try
        for z = 1:10
            [Mdl, tf] = ocsvm(X, PredictorNames=sX,ContaminationFraction=p1, KernelScale=p2, Lambda=p3, NumExpansionDimensions=p4, ...
                StandardizeData=p5, BetaTolerance=p6, ...
                GradientTolerance=p7, IterationLimit=p8);
            outliersSet = [outliersSet;tf'];
        end
        csvwrite(labelFile,outliersSet); 
    catch
        fprintf("-Failed\n")
    end
end