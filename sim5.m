close all    
clear        
clc          

diary('simulation_log.txt');

function mcs = getAdaptiveMCS(sinr)
    % Define MCS mapping based on SINR thresholds
    if sinr > 20
        % Good channel conditions
        mcs = 12;  % Example: QAM 64 with high coding rate
    elseif sinr > 15
        % Slightly lower but still good conditions
        mcs = 11;  % Example: QAM 64 with moderate coding rate
    elseif sinr > 10
        % Medium channel conditions
        mcs = randi([7, 10]);  % Randomly select from QAM 16 to QAM 64
    elseif sinr > 5
        % Poor channel conditions but still usable
        mcs = randi([3, 6]);    % Randomly select from QPSK to QAM 16
    else
        % Very poor channel conditions
        mcs = 2;                % Example: QPSK with low coding rate
    end
    
    % Optional: Add error handling for unexpected SINR values
    if isnan(sinr) || sinr < -10 || sinr > 30
        error('Invalid SINR value: %f', sinr);
    end
end

% Function to calculate path loss with shadowing and fast fading
function [pathLoss, shadowingLoss, fadingLoss] = calculateChannelLoss(distance, shadowingStdDev, fastFadingType)
    % Basic path loss (simplified free space)
    pathLoss = 32.4 + 20*log10(distance) + 20*log10(5.9);  % 5.9 GHz frequency
    
    % Log-normal shadowing
    shadowingLoss = shadowingStdDev * randn();
    
    % Fast fading
    if strcmp(fastFadingType, 'Rayleigh')
        fadingLoss = -10*log10(exprnd(1));
    else  % Rician
        fadingLoss = -10*log10(ricernd(ricianKFactor, 1));
    end
end

try
    % Parameters remain the same...
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
    BandMHz = 20;               % Bandwidth in MHz

    % Checkpointing setup
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

    % Set up parallel pool
    if isempty(gcp('nocreate'))
        parpool('local', 10);
    end

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
            
            % Save progress (using parallel safe file writing)
            saveProgress(trial, prr_trial);
            
        catch trialError
            % Log errors
            errorTime = datestr(now);
            errorMsg = sprintf('[%s] Error in trial %d: %s\n', errorTime, trial, getReport(trialError, 'extended'));
            fprintf(2, '%s', errorMsg);
            
            % Save error (using parallel safe file writing)
            saveError(errorMsg);
        end
    end

    % Analysis and visualization remain the same...
    PRR_results = gather(PRR_results);
    save('final_results.mat', 'PRR_results', 'rhoValues');

catch mainError
    errorTime = datestr(now);
    errorMsg = sprintf('[%s] Major simulation error: %s\n', errorTime, getReport(mainError, 'extended'));
    fprintf(2, '%s', errorMsg);
    saveError(errorMsg);
end

diary off;
delete(gcp('nocreate'));

% Helper function for parallel-safe progress saving
function saveProgress(trial, prr_trial)
    % Use file lock for safe parallel writing
    lockfile = 'progress.lock';
    while exist(lockfile, 'file')
        pause(0.1);
    end
    
    % Create lock
    fclose(fopen(lockfile, 'w'));
    
    try
        % Load existing data
        if exist('simulation_checkpoint.mat', 'file')
            data = load('simulation_checkpoint.mat');
            PRR_results = data.PRR_results;
        end
        
        % Update and save
        PRR_results(trial, :) = prr_trial;
        lastCompletedTrial = trial;
        save('simulation_checkpoint.mat', 'PRR_results', 'lastCompletedTrial');
    catch
        % Handle save error
    end
    
    % Remove lock
    delete(lockfile);
end

% Helper function for parallel-safe error logging
function saveError(errorMsg)
    % Use file lock for safe parallel writing
    lockfile = 'error.lock';
    while exist(lockfile, 'file')
        pause(0.1);
    end
    
    % Create lock
    fclose(fopen(lockfile, 'w'));
    
    try
        % Append error to log
        fid = fopen('error_log.txt', 'a');
        fprintf(fid, '%s', errorMsg);
        fclose(fid);
    catch
        % Handle write error
    end
    
    % Remove lock
    delete(lockfile);
end