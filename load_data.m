%function [targetNames, xTarget, yTarget, a, b, xEllipse, yEllipse, timeEllipse, capUCAV, bearingUCAV] = load_data()
%LOAD_DATA Summary of this function goes here
%   Detailed explanation goes here

    disp('Loading data, please wait a few minutes...');
    tic;

    scenario = 'XXXXXXXXXXX';
    targetsFolder = string(strcat('XXXXXXXXXXX/', scenario, '/XXXXXXXXXXX'));
    k = dir(char(strcat(targetsFolder, '\*.csv')));
    filenames = {k.name}';
    targetNumber = length(filenames);

    % Variables initialisation
    timeEllipse = cell(1,targetNumber);
    a = cell(1,targetNumber);
    b = cell(1,targetNumber);
    bearingUCAV = cell(1,targetNumber);
    xEllipse = cell(1,targetNumber);
    yEllipse = cell(1,targetNumber);
    xTarget = cell(1,targetNumber);
    yTarget = cell(1,targetNumber);
    targetNames = cell(1,targetNumber);
    
    for i = 1:targetNumber
        disp('Target n°')
        disp(i)
        
        % Extraction from Excel
        targetFile = char(strcat(targetsFolder, '/', filenames{i}));
        Excel_A = round(xlsread(targetFile, 'A:A')); % Time round to the nearest second
        Excel_NAA = xlsread(targetFile, 'N:AA');
        Excel_NAA(:, [4 5 6 7 10 11 12]) = []; % Remove useless columns

        M = cat(2, Excel_A, Excel_NAA); % All relevant infos from the current excel file

        % Remove rows of M which contain at least one zero in columns 2 3 4 5 6 (a, b, bearing, latEllipse, longEllipse)
        Mreduced = M(:,[2 3 4 5 6]); % Infos about ellipses
        M = M(all(Mreduced,2),:);

        % Remove duplicated ellipses
        Mreduced = M(:,[2 3 4 5 6]); % Need to update Mreduced because M has changed, maybe remove 4 ?
        [C,ia,ic] = unique(Mreduced, 'rows', 'stable'); % 'stable' keeps the row order unchanged
        M = M(ia,:); % ia corresponds to the unique row numbers in Mreduced

        
        %disp(M)
        
        % Outputs
        timeEllipse{i} = M(:,1);
        a{i} = M(:,2);
        b{i} = M(:,3);
        bearingUCAV{i} = M(:,4);
        [x,y,utmzone] = deg2utm(M(:,5), M(:,6));
        xEllipse{i} = correctEasting(x,utmzone);
        yEllipse{i} = y;
        [x,y,utmzone] = deg2utm(M(1,7), M(1,8));
        xTarget{i} = correctEasting(x,utmzone);
        yTarget{i} = y;
        targetNames{i} = filenames{i}(end-23:end-18);

    end

    Excel_FI = xlsread(strcat('XXXXXXXXXXX/', scenario, '/XXXXXXXXXXX.csv'), 'B:F');
    [xUCAV,yUCAV,utmzone] = deg2utm(Excel_FI(:,1), Excel_FI(:,2));
    xUCAV = correctEasting(xUCAV,utmzone);
    capUCAV = Excel_FI(:,5);
    
    clearvars -except timeEllipse a b bearingUCAV xEllipse yEllipse xTarget yTarget targetNames capUCAV xUCAV yUCAV;
    dataLoaded = 1;
    disp(['Data loaded in ' num2str(toc) ' seconds.'])
    
%end

function [correctedEasting] = correctEasting(easting,utmZone)
% All targets of the RECCE scenario are located in UTM zone 31T
% except SRSAM3 which is located in UTM zone 32T.
% Also UCAV location is in zone 32T at the beginning of the scenario.
% We'll need to manually correct the easting UTM coordinate when necessary.
    
    eastingMax31T = 489750; % approximative value of maximum easting in 31T
    indices32T = (utmZone(:,2)=='2'); % Get indices where utmZone='32 T'
    easting(indices32T) = easting(indices32T) + eastingMax31T;
    
    eastingMax32T = 489750*2; % approximative value of maximum easting in 32T
    indices33T = (utmZone(:,2)=='3'); % Get indices where utmZone='33 T'
    easting(indices33T) = easting(indices33T) + eastingMax32T;
    
    correctedEasting = easting;
end

