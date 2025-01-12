% Main script
close all    
clear        
clc          

diary('simulation_log.txt');

logFile = 'simulation_error_log.txt';
if ~exist('logs', 'dir')
   mkdir('logs');
end

% Add this at the start of your script
function logSystemInfo(logFile)
    try
        fid = fopen(logFile, 'a');
        fprintf(fid, '\n=== System Information ===\n');
        fprintf(fid, 'MATLAB Version: %s\n', version);
        fprintf(fid, 'Computer Name: %s\n', computer);
        fprintf(fid, 'Operating System: %s\n', computer('arch'));
        fprintf(fid, 'Available Memory: %.2f GB\n', memory.MemAvailableAllArrays/1e9);
        fprintf(fid, 'Number of Cores: %d\n', feature('numcores'));
        fprintf(fid, 'Current Directory: %s\n', pwd);
        fprintf(fid, '========================\n\n');
        fclose(fid);
    catch
        warning('Failed to log system information');
    end
end

% Call this at script start
logSystemInfo(fullfile('logs', 'simulation_error_log.txt'));

try
    packetSize = 500;          
    nTransm = 1;                % Number of transmissions per packet
    sizeSubchannel = 50;        % Number of Resource Blocks for each subchannel
    Raw = [50, 150, 300];       % Range of Awareness for evaluation of metrics
    speed = 50;                 % Average speed [km/h]
    speedStDev = 3;            % Speed standard deviation
    maxSpeedVar = 2;           % Maximum speed variation
    SCS = 15;                   % Subcarrier spacing [kHz]
    pKeep = 0.4;                % Keep probability for resource re-selection
    periodicity = 0.1;          % Generation interval (every 100 ms)
    sensingThreshold = -70;    % Threshold to detect resources as busy
    roadLength = 10000;          % Length of the road [m]
    configFile = 'Highway3GPP.cfg';

    % Channel parameters
    shadowingStdDev = 8;        % Shadowing standard deviation in dB
    fastFadingType = 'Rayleigh';% Fast fading type
    ricianKFactor = 3;          % Rician K-factor for LOS scenarios
    correlationDistance = 50;   % Correlation distance for shadowing 

    % Monte Carlo parameters
    numTrials = 10000;          % Number of Monte Carlo trials
    rhoValues = [100, 200, 300]; % Vehicle densities
    BandMHz = 20; 
    
    % Initialize or load checkpoint
    checkpointFile = 'simulation_checkpoint.mat';
    if exist(checkpointFile, 'file')
        checkpoint = load(checkpointFile);
        startTrial = checkpoint.lastCompletedTrial + 1;
        PRR_results = checkpoint.PRR_results;
        fprintf('Resuming from trial %d\n', startTrial);
    else
        startTrial = 1;
        PRR_results = zeros(numTrials, length(rhoValues));
        fprintf('Starting new simulation\n');
    end

    % Setting up parallel pool
    poolObj = gcp('nocreate');
    if isempty(poolObj)
        poolObj = parpool('local', 10);
    end
    cleanupObj = onCleanup(@() delete(poolObj)); % Will delete pool even if script crashes

    % Create a parallel.pool.DataQueue for progress updates
    dataQueue = parallel.pool.DataQueue;
    afterEach(dataQueue, @(data) updateProgress(data, checkpointFile));

    % Parallel processing for trials
    parfor trial = startTrial:numTrials
        try
            fprintf('Running trial %d of %d\n', trial, numTrials);
            
            % Initialize results for this trial
            prr_trial = zeros(1, length(rhoValues));
            
            % Create organized output directory
            trialFolder = fullfile(pwd, 'MCSTrials2k', sprintf('MonteCarloTrial_%04d', trial));
            if ~exist(trialFolder, 'dir')
                mkdir(trialFolder);
            end
            
            % Process each density sequentially within the trial
            for rhoIdx = 1:length(rhoValues)
                rho = rhoValues(rhoIdx);
                
                % Simulation time based on density
                switch rho
                    case 100
                        simTime = 10;
                    case 200
                        simTime = 5;
                    case 300
                        simTime = 3;
                end
                
                % SINR and MCS calculation
                distance = rand() * roadLength;
                [pathLoss, shadowingLoss, fadingLoss] = calculateChannelLoss(distance, shadowingStdDev, fastFadingType);
                sinr = 20 - pathLoss - shadowingLoss - fadingLoss;
                sinr = max(min(sinr, 30), -10);
                MCS = getAdaptiveMCS(sinr);
                
                % Speed calculation
                currentSpeed = max(0, speed + speedStDev * randn() + maxSpeedVar * sin(2 * pi * rand()));
                
                % Organized output folder for this density
                outputFolder = fullfile(trialFolder, sprintf('NRV2X_%dMHz_rho%d', BandMHz, rho));
                if ~exist(outputFolder, 'dir')
                    mkdir(outputFolder);
                end
                
                % Run simulation
                WiLabV2Xsim(configFile, 'outputFolder', outputFolder, 'Technology', '5G-V2X', ...
                    'MCS_NR', MCS, 'SCS_NR', SCS, 'beaconSizeBytes', packetSize, ...
                    'simulationTime', simTime, 'rho', rho, 'probResKeep', pKeep, ...
                    'BwMHz', BandMHz, 'vMean', currentSpeed, 'vStDev', speedStDev, ...
                    'cv2xNumberOfReplicasMax', nTransm, 'allocationPeriod', periodicity, ...
                    'sizeSubchannel', sizeSubchannel, 'powerThresholdAutonomous', sensingThreshold, ...
                    'Raw', Raw, 'FixedPdensity', false, 'dcc_active', true, 'cbrActive', true, ...
                    'roadLength', roadLength, 'channelModel', 0);
                
                % Process results
                prrFiles = dir(fullfile(outputFolder, 'packet_reception_ratio_*_5G.xls'));
                if ~isempty(prrFiles)
                    prrFile = fullfile(outputFolder, prrFiles(1).name);
                    if isfile(prrFile)
                        data = load(prrFile);
                        prr_trial(rhoIdx) = mean(data(:, end));
                    else
                        warning('File not found: %s. Skipping this density.', prrFile);
                        prr_trial(rhoIdx) = NaN;
                    end
                else
                    prr_trial(rhoIdx) = NaN;
                end
            end
            
            % Store results
            PRR_results(trial, :) = prr_trial;
            
            % Instead of calling saveProgress directly, send data to queue
            send(dataQueue, struct('trial', trial, 'prr_trial', prr_trial));
            
        catch trialError
            errorTime = datestr(now);
            errorMsg = sprintf('[%s] Error in trial %d: %s\n', errorTime, trial, getReport(trialError, 'extended'));
            fprintf(2, '%s', errorMsg);
            writeError(errorMsg);  % Use new error writing function
            logError(trialError, fullfile('logs', 'simulation_error_log.txt'), ...
                sprintf('Trial %d (Parallel Worker)', trial));
            rethrow(trialError);
        end
    end

    % Final results saving
    PRR_results = gather(PRR_results);
    save('final_results.mat', 'PRR_results', 'rhoValues');

catch mainError
    errorTime = datetime("now");
    errorMsg = sprintf('[%s] Major simulation error: %s\n', errorTime, getReport(mainError, 'extended'));
    fprintf(2, '%s', errorMsg);
    writeError(errorMsg);
    logError(mainError, fullfile('logs', 'simulation_error_log.txt'), 'Main Thread');
    rethrow(mainError);
end

diary off;
delete(gcp('nocreate'));

% Helper Functions (at the end of the file)
function mcs = getAdaptiveMCS(sinr)
    if sinr > 20
        mcs = 12;
    elseif sinr > 15
        mcs = 11;
    elseif sinr > 10
        mcs = randi([7, 10]);
    elseif sinr > 5
        mcs = randi([3, 6]);
    else
        mcs = 2;
    end
    
    if isnan(sinr) || sinr < -10 || sinr > 30
        error('Invalid SINR value: %f', sinr);
    end
end

% Create a function for structured error logging
function logError(errorInfo, logFile, context)
    % Get current timestamp
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');
    
    % Create a lock file path
    [logPath, logName, logExt] = fileparts(logFile);
    lockFile = fullfile(logPath, [logName '_lock' logExt]);
    
    % Try to acquire lock with timeout
    lockTimeout = 5; % seconds
    startTime = tic;
    while exist(lockFile, 'file')
        if toc(startTime) > lockTimeout
            warning('Could not acquire log file lock. Error may not be logged.');
            return;
        end
        pause(0.1);
    end
    
    try
        % Create lock file
        fid = fopen(lockFile, 'w');
        fclose(fid);
        
        % Open log file in append mode
        fid = fopen(logFile, 'a');
        
        % Write structured error information
        fprintf(fid, '\n%s\n', repmat('=', 1, 80));
        fprintf(fid, 'Timestamp: %s\n', timestamp);
        fprintf(fid, 'Context: %s\n', context);
        
        if isstruct(errorInfo)
            % For errors caught in try-catch
            fprintf(fid, 'Error Message: %s\n', errorInfo.message);
            fprintf(fid, 'Error Identifier: %s\n', errorInfo.identifier);
            
            % Print stack trace
            fprintf(fid, '\nStack Trace:\n');
            for i = 1:length(errorInfo.stack)
                fprintf(fid, '  File: %s\n', errorInfo.stack(i).file);
                fprintf(fid, '  Line: %d\n', errorInfo.stack(i).line);
                fprintf(fid, '  Function: %s\n\n', errorInfo.stack(i).name);
            end
        else
            % For string error messages
            fprintf(fid, 'Error Message: %s\n', errorInfo);
        end
        
        fprintf(fid, '%s\n', repmat('=', 1, 80));
        
        % Close file
        fclose(fid);
    catch logError
        warning('Failed to write to error log: %s', E.logError.message);
    end
    
    % Remove lock file
    if exist(lockFile, 'file')
        delete(lockFile);
    end
end

% Improve the channel loss calculation function
function [pathLoss, shadowingLoss, fadingLoss] = calculateChannelLoss(distance, shadowingStdDev, fastFadingType, ricianKFactor)
    if distance <= 0 || shadowingStdDev < 0
        error('Invalid distance or shadowing parameters');
    end
    
    pathLoss = 32.4 + 20*log10(distance) + 20*log10(5.9);
    shadowingLoss = shadowingStdDev * randn();
    
    switch fastFadingType
        case 'Rayleigh'
            fadingLoss = -10*log10(exprnd(1));
        case 'Rician'
            if nargin < 4 || isempty(ricianKFactor)
                error('Rician K-factor must be provided for Rician fading');
            end
            fadingLoss = -10*log10(ricernd(ricianKFactor, 1));
        otherwise
            error('Unsupported fading type: %s', fastFadingType);
    end
end

% Improve the progress update mechanism with file locking
function updateProgress(data, checkpointFile)
    lockFile = [checkpointFile '.lock'];
    while ~mkdir(lockFile) % Try to create lock directory
        pause(0.1); % Wait before retrying
    end
    try
        if exist(checkpointFile, 'file')
            checkpoint = load(checkpointFile);
            PRR_results = checkpoint.PRR_results;
        end
        
        PRR_results(data.trial, :) = data.prr_trial;
        lastCompletedTrial = data.trial;
        save(checkpointFile, 'PRR_results', 'lastCompletedTrial');
    catch
        Print("----> Some error has happened when updating the progress <----")
    finally
        rmdir(lockFile); % Release lock
    end
end

function writeError(errorMsg)
    % Simple file append with built-in locking
    fid = fopen('error_log.txt', 'a');
    if fid ~= -1
        fprintf(fid, '%s', errorMsg);
        fclose(fid);
    end
end

