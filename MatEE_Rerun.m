clear
clc
%% Init
main_function()
%% Main Function
function main_function()
    filename = 'GD_ReRun/MatEE.csv';
    opts = detectImportOptions(filename);
    opts = setvartype(opts,'char');  % or 'string'
    T = readtable(filename,opts);
    T = table2array(T);
    for i = 1:size(T,1)
        parameters = [];
        Method.name = "Method";
        Method.default = cell2mat(T(i,2));
        OutlierFraction.name = "OutlierFraction";% fcmd, olivehawkins
        OutlierFraction.default = str2num(cell2mat(T(i,3)));
        NumTrials.name = "NumTrials"; % 500 if fcmd, 2 if olivehawkins
        NumTrials.default = str2num(cell2mat(T(i,4)));
        BiasCorrection.name = "BiasCorrection"; %fcmd
        BiasCorrection.default = str2num(cell2mat(T(i,5)));
        NumOGKIterations.name = "NumOGKIterations"; % ogk
        NumOGKIterations.default = str2num(cell2mat(T(i,6)));
        UnivariateEstimator.name = "UnivariateEstimator"; % ogk
        UnivariateEstimator.default = cell2mat(T(i,7));
        ReweightingMethod.name = "ReweightingMethod";%olivehawkins
        ReweightingMethod.default = cell2mat(T(i,8));
        NumConcentrationSteps.name = "NumConcentrationSteps";%olivehawkins
        NumConcentrationSteps.default = str2num(cell2mat(T(i,9)));
        StartMethod.name = "StartMethod";%olivehawkins
        StartMethod.default = cell2mat(T(i,10));

        parameters = [Method, OutlierFraction, NumTrials,BiasCorrection,NumOGKIterations,UnivariateEstimator,ReweightingMethod,NumConcentrationSteps,StartMethod];
    
        
        EE(cell2mat(T(i,1)), parameters);
        

        
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

%% EE
function EE(filename, parameters)
    readfilename = sprintf('Dataset/%s', filename);
    
    if isfile(sprintf('Dataset/%s.csv', filename))
        [X, y] = csvfileread(sprintf('Dataset/%s.csv', filename));
    end
    if isfile(sprintf('Dataset/%s.mat', filename))
        [X, y] = matfileread(sprintf('Dataset/%s.mat', filename));
        
    end
    
    runEE(filename, X, y, parameters);
            
  end
%% Run EE
function runEE(filename_with_extension, X, y, params)
    filename_char = convertStringsToChars(filename_with_extension);
    filename = filename_char;
    labelFile = "Labels/EE_Matlab/Labels_Mat_EE_"+filename + "_" + params(1).default + "_" + params(2).default + "_" + params(3).default + "_" + params(4).default + "_" + params(5).default + "_" + params(6).default + "_" + params(7).default + "_" + params(8).default + "_" + params(9).default + ".csv";
    if isfile(labelFile)
       return
    end
    p1 = params(1).default;
    p2 = params(2).default;
    p3 = params(3).default;
    p4 = params(4).default;
    p5 = params(5).default;
    p6 = params(6).default;
    p7 = params(7).default;
    p8 = params(8).default;
    p9 = params(9).default;
    
    outliersSet = [];
    try
        for z = 1:10
            if strcmp(p1, "fmcd") == 1
                [sig,mu,mah,outliers] = robustcov(X, Method=p1, OutlierFraction=p2, NumTrials=p3, BiasCorrection=p4);
        
            elseif strcmp(p1, "ogk") == 1
                [sig,mu,mah,outliers] = robustcov(X, Method=p1, NumOGKIterations=p5, UnivariateEstimator=p6);
            elseif strcmp(p1, "olivehawkins") == 1
                [sig,mu,mah,outliers] = robustcov(X, Method=p1, OutlierFraction=p2, ...
                    ReweightingMethod=p7, NumConcentrationSteps=p8, StartMethod=p9);
            end
            outliersSet = [outliersSet;outliers'];
        end
        csvwrite(labelFile,outliersSet); 
    catch
        fprintf("-Failed")
    end
end