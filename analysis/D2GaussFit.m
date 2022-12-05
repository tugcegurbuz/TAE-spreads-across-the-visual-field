%% Fitting 2D Gaussian to human TAE data
% Requirements:
% - human_data.mat
% - mycolormap2.mat %for color map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; % memory
close all; % windows

%% ---Fitting Functions---
%
% Coeficients A convention:
%	A = [Amplitude, x0, x-Width, y0, y-Width, Angle(in Radians)]
%
% X-data convention:
%	X is of size(n,n,2) where 
%	X(:,:,1) : x-coordinates,  
%	X(:,:,2) : y-coordinates.
%
% In this numerical test we use two-dimensional fitting functions:
% 1. 2D Gaussian function ( A requires 5 coefs ).
g = @(A,X) A(1)*exp( -((X(:,:,1)-A(2)).^2/(2*A(3)^2) + (X(:,:,2)-A(4)).^2/(2*A(5)^2)) );
% 2. 2D Rotated Gaussian function ( A requires 6 coefs ).
f = @(A,X) A(1)*exp( -(...
    ( X(:,:,1)*cos(A(6))-X(:,:,2)*sin(A(6)) - A(2)*cos(A(6))+A(4)*sin(A(6)) ).^2/(2*A(3)^2) + ... 
    ( X(:,:,1)*sin(A(6))+X(:,:,2)*cos(A(6)) - A(2)*sin(A(6))-A(4)*cos(A(6)) ).^2/(2*A(5)^2) ) );

%% ---Data---
% Get x, y, z data from the file
% The data is in the form of a matrix with 3 columns
data = load('human_data.mat').DATA;
x = data(:,1);
y = data(:,2);
z = data(:,3);
% Perform interpolation
% The data is interpolated to a regular grid
xq = linspace(min(x),max(x),100);
yq = linspace(min(y),max(y),100);
[Xq,Yq] = meshgrid(xq,yq);
Zq = griddata(x,y,z,Xq,Yq,'natural');
% Prepare the data for the fitting function
X = zeros(size(Xq,1),size(Xq,2),2);
X(:,:,1) = Xq;
X(:,:,2) = Yq;
S = Zq;


%% ---Parameters---
A0 = [8, 10.5, 5, 0, 3, 0];   % Inital (guess) parameters
InterpMethod='natural'; % 'nearest','linear','spline','cubic'
MaxIter=1000;			% Maximum number of iterations

%% ---Fit---
% Define lower and upper bounds [Amp,xo,wx,yo,wy,fi]
LB = [8, 8, 1, 0, 1, -pi];
UB = [9, 10.5, 8, 1, 5, pi];
% Fit
[A,RESNORM,RESIDUAL,EXITFLAG,OUTPUT,LAMBDA,JACOBIAN] = lsqcurvefit(f,A0,X,S,LB,UB, ...
    optimset('MaxIter',MaxIter));
disp(OUTPUT); % display summary of LSQ algorithm

%% ---Plot Data---
load('./mycolormap2.mat'); % load colormap
% Plot the original data
figure(1);
surf(Xq,Yq,Zq);
shading interp

colormap(mymap)
cbar = colorbar;
cbar.Location = 'northoutside';
cbar.Label.String = 'TAE Magnitude';
caxis([0 10])
brighten(.3)

xlim([0 20])
names_x = {'Near1', 'Near2', 'Center', 'Further1', 'Further2'};
set(gca,'xtick',[2.5:4:20.5],'xticklabel',names_x)
ylim([-8, 8])
names_y = {'', 'down (d)', 'center (c)', 'up (u)', ''};
set(gca,'ytick',[-8:4:8],'yticklabel',names_y, 'FontSize', 20)

grid on
ax.GridAlpha = 1;

hold on

% Plot fitted model as a contour
Z_fit = f(A,X);
[M,c] = contour3(Xq,Yq,Z_fit+10,8);
c.LineWidth = 2;
c.LineColor = [0.1 0.1 0.1];

% Add model parameters to the plot (show 2 decimal)
text(16.5, -4.5, 0.5, ['Amp = ' num2str(A(1), '%.2f')], 'FontSize', 20)
text(16.5, -5, 0.5, ['x0, y0 = ' num2str(A(2), '%.2f') ',' num2str(A(4), '%.2f')], 'FontSize', 20)
text(16.5, -5.5, 0.5, ['wx, wy = ' num2str(A(3), '%.2f') ',' num2str(A(5), '%.2f')], 'FontSize', 20)
text(16.5, -6, 0.5, ['Ori = ' num2str(A(6), '%.2f')], 'FontSize', 20)

