clear
clc
%% Init
main_function()
%% Main Function
function main_function()
    filename = 'GD_ReRun/MatIF.csv';
    opts = detectImportOptions(filename);
    opts = setvartype(opts,'char');  % or 'string'
    T = readtable(filename,opts);
    T = table2array(T);
    for i = 1:size(T,1)
        
        parameters = [];
        ContaminationFraction.name = "ContaminationFraction";
        ContaminationFraction.default = cell2mat(T(i,2));
    
        NumLearners.name = "NumLearners";
        NumLearners.default = cell2mat(T(i,3));
    
        NumObservationsPerLearner.name = "NumObservationsPerLearner";
        NumObservationsPerLearner.default = cell2mat(T(i,4));
    
        
        parameters = [ContaminationFraction, NumLearners, NumObservationsPerLearner];
    
        
        IF(cell2mat(T(i,1)), parameters);
        
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

%% IF
function IF(filename, parameters)
    readfilename = sprintf('Dataset/%s', filename);
    
    if isfile(sprintf('Dataset/%s.csv', filename))
        [X, y] = csvfileread(sprintf('Dataset/%s.csv', filename));
    end
    if isfile(sprintf('Dataset/%s.mat', filename))
        [X, y] = matfileread(sprintf('Dataset/%s.mat', filename));
        
    end

    runIF(filename, X, y, parameters);
            
  end
%% Run IF
function runIF(filename_with_extension, X, y, params)
    filename_char = convertStringsToChars(filename_with_extension);
    filename = filename_char;
    labelFile = "Labels/IF_Matlab/Labels_Mat_IF_"+filename + "_" + params(1).default + "_" + params(2).default + "_" + params(3).default + ".csv";
    
    if isfile(labelFile)
        return
    end

    
    p1 = params(1).default;
    p2 = params(2).default;
    p3 = params(3).default;
    
    if string(p1) == "LOF"
        percentage_table = readtable("Stats/SkPercentage.csv");
        percentage_table_file = percentage_table(string(percentage_table.Filename)==filename, :);
        p1 = percentage_table_file.LOF;
    elseif string(p1) == "IF"
        percentage_table = readtable("Stats/SkPercentage.csv");
        filename_char = convertStringsToChars(filename);
        percentage_table_file = percentage_table(string(percentage_table.Filename)==filename, :);
        p1 = percentage_table_file.IF;
    else
        p1 = str2double(p1);
    end
    if string(p3) == "auto"
        p3 = min(size(X,1), 256);
    else
        p3 = floor(str2double(p3)*size(X,1));
    end
    outliersSet = [];
%     try
        for z = 1:10
            [forest, tf, score] = iforest(X, ContaminationFraction=p1, NumLearners=str2double(p2), NumObservationsPerLearner=p3);
            outliersSet = [outliersSet;tf'];

        end
        csvwrite(labelFile,outliersSet); 
%     catch
%         fprintf("-Failed")
%     end
end