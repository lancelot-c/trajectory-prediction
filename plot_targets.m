
% Load data if it's not already loaded
if exist('dataLoaded','var') == 0
    run('load_data');
end
  
% Display each target on a different plot but on the same figure
figure('Name','SAR images with coeff = 1.5','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
for target = 1:length(a)
    
    subplot(2,4,target);
    %title('RECCE scenario in the UTM coordinate system');
    %xlabel('Easting (km)');
    %ylabel('Northing (km)');
    daspect([1 1 1]);
    hold on;


    
    % Plot ellipses from the target 
    theta = 0 : 0.01 : 2*pi;
    ellipsesNumber = length(a{target});
    
    for i = 1:ellipsesNumber
        
        % Axes
        x = a{target}(i)/2 .* cos(theta);
        y = b{target}(i)/2 .* sin(theta);
        
        % Rotation + Translation towards ellipse center
        currentBearing = bearingUCAV{target}(i);
        currentCap = capUCAV(timeEllipse{target}(i)+1); % +1 because time starts at 0 in UCAV file
        r = 90 - currentCap - currentBearing; 
        xr = x*cosd(r)-y*sind(r) + xEllipse{target}(i);
        yr = x*sind(r)+y*cosd(r) + yEllipse{target}(i);
        
        % Plot
        startRate = 0.95; % blanc = 1, noir = 0
        endRate = 0;
        colorStep = (startRate-endRate)/(ellipsesNumber-1); % ellipsesNumber must be >= 2
        colorRate = startRate-(i-1)*colorStep; % colorRate = 1-(i/ellipsesNumber);
        p2 = plot(xr/1000, yr/1000, 'Color', [colorRate colorRate colorRate]);
    end

    % SAR image spanning the last ellipse
    coeffSAR = 2.0;
    wSAR = (a{target}(ellipsesNumber)/1000)*coeffSAR;
    hSAR = wSAR;
    xc = xEllipse{target}(ellipsesNumber)/1000;
    yc = yEllipse{target}(ellipsesNumber)/1000;
    
    xSAR = [xc-(wSAR/2), xc-(wSAR/2), xc+(wSAR/2), xc+(wSAR/2), xc-(wSAR/2)];
    ySAR = [yc-(hSAR/2), yc+(hSAR/2), yc+(hSAR/2), yc-(hSAR/2), yc-(hSAR/2)];
    
    xSAR = xSAR - xc;
    ySAR = ySAR - yc;
    
    xrSAR = xSAR*cosd(r)-ySAR*sind(r) + xc;
    yrSAR = xSAR*sind(r)+ySAR*cosd(r) + yc;
    
    p3 = plot(xrSAR, yrSAR, 'Color', [0.14 0.4 0]);
    
    
    % Plot the true target position
    % /1000 for meters to kilometers conversion
    p1 = scatter(xTarget{target}/1000,yTarget{target}/1000, '+', 'MarkerEdgeColor', [0.8 0.1 0.1]);
    text(xTarget{target}/1000, yTarget{target}/1000, [' ' targetNames{target}]);
    hold off;
   
    
    
    % Add legend
    %legend([p1 p2 p3], 'True target location', 'Estimated target location', 'SAR image', 'Location','southwest');
    %legend('boxoff');
end




