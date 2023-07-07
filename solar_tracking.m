%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function solar_tracking()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script tests a variety of solar insolations in FISEBM.
% var1 ~ epsilon (emissivity of the atmosphere)
% var2 ~ ice flow condition ('flow' or 'no-flow')
% var3 ~ Nstart (the model iteration number, 1 for a new run, else a restart run) 
% var4 ~ Q (solar constant)
% var5 ~ expnum (corresponding to the initial climate state, i.e. globally glaciated, global ocean, etc.)

%%%% Example %%%%
% Option: epsilon
var1 = 0.95;
% Option: flow or no-flow
var2 = 'flow';
% Option: Nstart
var3 = 1;
% Option: Range of Solar Insolation
Q = 900;
% Option: expnum
var5 = 3;

% Option: Parallel Computing - FISEBM Model
if length(Q)==1
  var4 = Q;
  FISEBM(var1,var2,var3,var4,var5);
else
  pool = parpool('local',3);

  parfor i = 1:length(Q)
    var4 = Q(i);
    FISEBM(var1,var2,var3,var4,var5);
  end
end

delete(pool)

fprintf(1,'solar is done.\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function FISEBM(var1,var2,var3,var4,var5)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Asynchronous Coupling of Floating Ice Sheets 1D Spherical Ice Flow Model
% with a version of the Pollard and Kasting, 2005 Energy Balance Model. 
% This version of FISEBM is reading epsilon restarts and saving the plots
% in the FISEBM folder (/epsilon/FISEBM/). 

% internal ice flow clock - 10000 years

addpath('/u/home/f/fspauldi/exosnowballs/ver02/Insolation');

% Conditions for Aborting Model
icelatlim = 0;
icelatpole = 0;
icelateq = 0;
icelatstable = 0;
noice = 0;

% Time
year = 365*86400;
t_stable = 0;
t_stable_max = 1e4*year;
Nstart = var3;
Nend = 100;

expnum = var5;

Qtest = var4;
Qo = var4;
Qo = Qo';
iceline_N = NaN(length(Qo),length(expnum));
iceline_S = NaN(length(Qo),length(expnum));

% Create Folder to Export Plots
plotfolder = '/u/scratch/f/fspauldi/exosnowballs/Figures';
if ~exist(sprintf('%s',plotfolder),'file')
  [status,msg] = mkdir(sprintf('%s',plotfolder));
end

% Create Folder to Export Restart Files
restartfolder = '/u/scratch/f/fspauldi/exosnowballs/FISEBMRestart';
if ~exist(sprintf('%s',restartfolder),'file')
  [status,msg] = mkdir(sprintf('%s',restartfolder));
end

for i = 1:length(Qo)
  for j = 1:length(expnum)
    EBM_expnum = expnum(j);


    for N = Nstart:Nend
      N

      EBMpar = EBM(EBM_expnum,N,Qo(i),var1,var2);

      %if EBMpar.icelatpole==1
      %iceline_N(i,j) = EBMpar.Nedge;
	    %iceline_S(i,j) = EBMpar.Sedge;
	    %continue;
      %end

      %keyboard;

      FISpar = fis_1D_sphere(EBM_expnum,N,Qo(i),var1,var2);

      if FISpar.noice==0 && FISpar.icelatpole==0

	%%% Plotting FIS ouput over Time %%%
	% Declare Initial Conditions
	if N==1
	  Rplot_EBM = EBMpar.R_initial;
	  Rplot_FIS = FISpar.R_initial;

	  hplot_ave = EBMpar.h_init_ave;
	  hplot_max = max(EBMpar.h_initial);
	  hplot_min = min(EBMpar.h_initial);

	  Toplot_ave = EBMpar.To_init_ave;
	  Toplot_max = max(EBMpar.To_initial);
	  Toplot_min = min(EBMpar.To_initial);

	  Taplot_ave = FISpar.Ta_init_ave;
	  Taplot_max = max(FISpar.Ta_initial);
	  Taplot_min = min(FISpar.Ta_initial);

	  vplot_ave = 0;
	  vplot_max = 0;
	  vplot_min = 0;

	  nplot_FIS = 0;

	  Eplot_toa = 0;
	  Eplot_sfc = 0;

	  netEvap = 100;
	  netPrec = 100;
	  Aplot = 0.30;

	  tplot = 0; 
	  masterclock = 0;
	elseif N==Nstart && Nstart>1
	  r_filename = sprintf('%s/restart-FISEBM-expnum-%.2d-Q-%.2d-N-%.2d-eps-%.2d-%s.mat',restartfolder,EBM_expnum,Qtest,N-1,100*var1,var2);
	  load(r_filename);
	  % all the above fields are inputed
	end
	
	dt = (FISpar.n/FISpar.nt)*FISpar.Time;
	masterclock = masterclock + dt;
	
	% Concatenate 1:N-1 to N 
	Rplot_EBM = horzcat(Rplot_EBM,EBMpar.R_end);
	Rplot_FIS = horzcat(Rplot_FIS,FISpar.R_end);

	hplot_ave = horzcat(hplot_ave,EBMpar.h_ave);
	hplot_max = horzcat(hplot_max,EBMpar.h_max);
	hplot_min = horzcat(hplot_min,EBMpar.h_min);

	Toplot_ave = horzcat(Toplot_ave,EBMpar.To_ave);
	Toplot_max = horzcat(Toplot_max,EBMpar.To_max);
	Toplot_min = horzcat(Toplot_min,EBMpar.To_min);

	Taplot_ave = horzcat(Taplot_ave,FISpar.Ta_ave);
	Taplot_max = horzcat(Taplot_max,FISpar.Ta_max);
	Taplot_min = horzcat(Taplot_min,FISpar.Ta_min);

	vplot_ave = horzcat(vplot_ave,FISpar.v_ave);
	vplot_max = horzcat(vplot_max,FISpar.v_max);
	vplot_min = horzcat(vplot_min,FISpar.v_min);

	nplot_FIS = horzcat(nplot_FIS,FISpar.n);
	
	Eplot_toa = horzcat(Eplot_toa,EBMpar.Eplot_toa);
	Eplot_sfc = horzcat(Eplot_sfc,EBMpar.Eplot_sfc);

	netPrec = horzcat(netPrec,EBMpar.netPrec);
	netEvap = horzcat(netEvap,EBMpar.netEvap);

	Aplot = horzcat(Aplot,EBMpar.Aplot);

	% Other Fields
	tplot = horzcat(tplot,masterclock);
	x1 = tplot/(FISpar.year*1000); % time in 10^3 yrs

	Nplot = [0:N];
	x2 = Nplot;

	%%%% Save Figure (1) %%%%
	m=5; n=1; pos = 2;
	clf;
	figure(1)
	fig = figure(1);
	set(fig,'Visible','off');

	% Number of time steps per ice flow run 
	subplot(m,n,pos-1);
	plot(x2,nplot_FIS);
	%h1=ylabel('latitude');
	%h2=xlabel('N');
	h3=title(sprintf('FIS #n per N'));

	% Ice Thickness over Time
	subplot(m,n,pos);
	plot(x2,hplot_ave,x2,hplot_max,x2,hplot_min); 
	%h1=ylabel('latitude');
	%h2=xlabel('N');
	h3=title(sprintf('h'));
	legend({'have','hmax','hmin'},'Position',[0.18 0.7 0.05 0.05],'FontSize',4);
	%set([gca,h3],'fontsize',5);

	% Ocean Temp over Time
	subplot(m,n,pos+1);
	plot(x1,Toplot_ave,x1,Toplot_max,x1,Toplot_min); 
	%h1=xlabel('latitude');
	%h2=xlabel('t');
	h3=title(sprintf('To'));
	legend({'Toave','Tomax','Tomin'},'Position',[0.18 0.53 0.05 0.05],'FontSize',4);
	%set([gca,h3],'fontsize',5);

	% Mean Atmos Temp over Time
	subplot(m,n,pos+2);
	plot(x1,Taplot_ave,x1,Taplot_max,x1,Taplot_min); 
	%h1=xlabel('latitude');
	%h2=xlabel('N');
	h3=title(sprintf('Ta'));
	legend({'Taave','Tamax','Tamin'},'Position',[0.18 0.355 0.05 0.05],'FontSize',4);
	%set([gca,h3],'fontsize',5);

	subplot(m,n,pos+3);
	plot(x1,vplot_ave,x1,vplot_max,x1,vplot_min);
	h3=title('v (m/yr)');
	h2 = xlabel('t (10^3)');
	legend({'vave','vmax','vmin'},'Position',[0.18 0.18 0.05 0.05],'FontSize',4);
	%set([gca,h2,h3],'fontsize',5);

	saveas(gcf,sprintf('%s/hovmoller-Q-%.2d-exp-%.2d-eps-%.2d-%s-part-1.png',plotfolder,Qtest,EBM_expnum,100*var1,var2));

	%%%% Save Figure (2) %%%%
	m=2; n=1;
	clf;
	figure(2)
	fig = figure(2);
	set(fig,'Visible','off');

	% Hovmoller of R over time
	subplot(m,n,1);
	jb=2; je=FISpar.nj-1;
	X = 90-FISpar.theta(je:-1:jb);
	Y = x1;
	Z = Rplot_EBM(je:-1:jb,:);
	contourf(X,Y,Z','LineColor','none');
	colormap(cool);
	colorbar('FontSize',4,'Position',[0.92 0.83 0.02 0.1]);
	h1=ylabel('t');
	%h2=xlabel('Latitude');
	h3=title(sprintf('R - EBM'));
	%set([gca,h1,h2,h3],'fontsize',5);

	% Hovmoller of R over time
	subplot(m,n,2);
	jb=2; je=FISpar.nj-1;
	X = 90-FISpar.theta(je:-1:jb);
	Y = x1;
	Z = Rplot_FIS(je:-1:jb,:);
	contourf(X,Y,Z','LineColor','none');
	colormap(cool);
	colorbar('FontSize',4,'Position',[0.92 0.23 0.02 0.1]);
	h1=ylabel('t');
	%h2=xlabel('Latitude');
	h3=title(sprintf('R - FIS'));
	%set([gca,h1,h2,h3],'fontsize',5);

	saveas(gcf,sprintf('%s/hovmoller-Q-%.2d-exp-%.2d-eps-%.2d-%s-part-2.png',plotfolder,Qtest,EBM_expnum,100*var1,var2));

	%%%% Save Figure (3) %%%%
	m=2; n=1;
	clf;
	figure(3)
	fig = figure(3);
	set(fig,'Visible','off');

	% ToA Energy Balance over N
	subplot(m,n,1);
	plot(x1,Eplot_toa);
	%h1=ylabel('latitude');
	h2=xlabel('t (10^3 yrs)');
	h3=title(sprintf('ToA Energy Balance = %.2d',Eplot_toa(end)));

	% Surface Energy Balance over N
	subplot(m,n,2);
	plot(x2,Eplot_sfc);
	%h1=ylabel('latitude');
	h2=xlabel('N');
	h3=title(sprintf('Surface Energy Balance = %.2d',Eplot_sfc(end)));

	saveas(gcf,sprintf('%s/hovmoller-Q-%.2d-exp-%.2d-eps-%.2d-%s-part-3.png',plotfolder,Qtest,EBM_expnum,100*var1,var2));

	%%%% Save Figure (4) %%%%
	m=2; n=1;
	clf;
	figure(4)
	fig = figure(4);
	set(fig,'Visible','off');

	% Net Prec & Evap over N
	subplot(m,n,1);
	plot(x1,netPrec,x1,netEvap);
	h2=xlabel('t (10^3 yrs)');
	h3=title(sprintf('Net Prec = %.2d & Net Evap = %.2d',netPrec(end)*EBMpar.year*100/EBMpar.rho_i,netEvap(end)*EBMpar.year*100/EBMpar.rho_i));

	% Global Mean Albedo
	subplot(m,n,2);
	plot(x2,Aplot);
	h2=xlabel('N');
	h3=title(sprintf('Global Mean Albedo = %.2d',Aplot(end)));

	saveas(gcf,sprintf('%s/hovmoller-Q-%.2d-exp-%.2d-eps-%.2d-%s-part-4.png',plotfolder,Qtest,EBM_expnum,100*var1,var2));
	

	% Iceline Stability reached after some time T
	%if FISpar.icelatstable==1
	  %t_stable = t_stable + FISpar.Time; 
	  %if t_stable >= t_stable_max
	    %if FISpar.Nedge==0
	      %FISpar.Nedge = FISpar.EQ;
	    %end
	    %if FISpar.Sedge==0
	      %FISpar.Sedge = FISpar.EQ;
	    %end
	    %iceline_N(i,j) = FISpar.Nedge;
	    %iceline_S(i,j) = FISpar.Sedge;
	    %continue;
	  %end  
	%end
      end

      if FISpar.icelateq==1
	if FISpar.Nedge==0
	  FISpar.Nedge = FISpar.EQ;
	end
	if FISpar.Sedge==0
	  FISpar.Sedge = FISpar.EQ;
	end
	iceline_N(i,j) = FISpar.Nedge;
	iceline_S(i,j) = FISpar.Sedge;
	continue;
      end
      
      if FISpar.noice==1 || FISpar.icelatpole==1
	if strcmp(FISpar.text,'start')==1
	  if FISpar.Nedge_initial==0
	    FISpar.Nedge_initial = FISpar.EQ;
	  end
	  if FISpar.Sedge_initial==0
	    FISpar.Sedge_initial = FISpar.EQ;
	  end
	  iceline_N(i,j) = FISpar.Nedge_initial;
	  iceline_S(i,j) = FISpar.Sedge_initial;
	  break;
	elseif strcmp(FISpar.text,'end')==1
	  if FISpar.Nedge==0
	    FISpar.Nedge = FISpar.EQ;
	  end
	  if FISpar.Sedge==0
	    FISpar.Sedge = FISpar.EQ;
	  end
	  iceline_N(i,j) = FISpar.Nedge;
	  iceline_S(i,j) = FISpar.Sedge;
	  break;
	end
      end

      save(sprintf('%s/restart-FISEBM-expnum-%.2d-Q-%.2d-N-%.2d-eps-%.2d-%s.mat',restartfolder,EBM_expnum,Qtest,N,100*var1,var2),'iceline_N','iceline_S','Rplot_EBM','Rplot_FIS','hplot_ave','hplot_max','hplot_min','Toplot_ave','Toplot_max','Toplot_min','Taplot_ave','Taplot_max','Taplot_min','tplot','masterclock','vplot_ave','vplot_max','vplot_min','nplot_FIS','Eplot_toa','Eplot_sfc','netEvap','netPrec','Aplot');

      if N>=20 && FISpar.n==FISpar.nt
        fprintf('Model Equilibrated: N>=20 & ice flow model reached 1Myr');
        break;
      end

    end

  % Rewrite the Restart after each test
  if FISpar.noice==0 && FISpar.icelatpole==0
  save(sprintf('%s/restart-FISEBM-expnum-%.2d-Q-%.2d-N-%.2d-eps-%.2d-%s.mat',restartfolder,EBM_expnum,Qtest,N,100*var1,var2),'iceline_N','iceline_S','Rplot_EBM','Rplot_FIS','hplot_ave','hplot_max','hplot_min','Toplot_ave','Toplot_max','Toplot_min','Taplot_ave','Taplot_max','Taplot_min','tplot','masterclock','vplot_ave','vplot_max','vplot_min','nplot_FIS','Eplot_toa','Eplot_sfc','netEvap','netPrec','Aplot');
  end

  end
end

fprintf(1,'FISEBM, done.\n');

% restart files saved for FIS & EBM as part of their subroutines
% export files will be the restart files in FIS & EBM
% import files will be called in read_restart - write seperate function
% restart files need to include the master time, N
% import files need to be established in both FIS & EBM 
% par.Hcr defined in both EBM & FIS Input files

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function par = fis_1D_sphere(EBM_expnum,N,Qo,var1,var2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve the model for Marine ice sheets
%% Eli, June 2011
%% Francisco, June 2017

par = set_FIS_parameters(EBM_expnum,N,Qo);

%% our 1d horizontal flow model using spherical coordinates:
par = integrate_h_1d_sphere(par,var1,var2);

fprintf(1,'done.\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function par=set_FIS_parameters(EBM_expnum,N,Qo)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

par.alpha_o = 0.25;
par.alpha_i = 0.6;

par.N = N;
par.EBM_expnum = EBM_expnum;
par.Nt = 500;
if par.N>5
  par.Nplot = 5;
else
  par.Nplot = 1;
end
par.Qo = Qo;

par.h_o = 50; %xx 30
par.c_o = 4218;
par.G = 0.06; % geothermal heat flux, W/m^2
par.k_i = 2.1; % ice thermal conductivity, W/(m)
par.D_o = 3e4; % ocean heat diffusivity xx8e5
par.Beta = 7; %ocean-ice base heat flux coefficient, W/(m^2 K)

par.icelatlim = 0;
par.icelatpole = 0;
par.icelateq = 0;
par.icelatstable = 0;
par.noice = 0;

%% initial thickness if no restart file exists:
par.h0=1000;

par.g=10;
par.T_f=273.15;
par.rho_o=1000;
par.rho_i=900;
%% (J/kg, from wikipedia's latent_heat), 
%% http://en.wikipedia.org/wiki/Latent_heat#Table_of_latent_heats :
par.L_f=334e3; 
par.mu=par.rho_i/par.rho_o;
par.year=365*86400;
par.S0=0.015/par.year; % m/sec
par.L=2e7;
par.R=6300e3; % Earth radius
par.D=1000;
% Thickness diffusivity, numerical only, make it as small as possible:
par.kappa=1.2e0;    % 1.0e0
par.kappa_2d=1.2e0;  % 1.0e0
par.T_surface_profile_type='warm';


par.Time=1e6*par.year; % time to run the model for (n=2,51*par.year)
%par.Time = 100*par.year;
par.Time_2d=1e5*par.year;
%par.dt=0.5e2*par.year;
%par.dt = par.year;
par.dt = par.year;
par.dt_2d=0.5e1*par.year;
par.nt=ceil(par.Time/par.dt); % par.nt = 2000
par.nt_2d=ceil(par.Time_2d/par.dt_2d);

par.nplot=200;
par.nplot_2d=100;
par.do_use_imagesc=0;
par.plot_2d.min_h=NaN;
par.plot_2d.max_h=NaN;
par.plot_2d.do_h_ylabel=1;
par.nwrite_restart=min(par.nplot,floor(par.nt/10));


par.ni=89;
par.nj=89;
par.nk=42;
par.EQ = ceil(par.nj/2);

%% file with E-P/ melt-freeze forcing:
par.S_filename=...
    sprintf('~/Floating-ice-sheet-dynamics/Input/Pollard/pollard_forcing_interpolated_nj=%3.3d.mat',par.nj);

%% physical domain, including boundary points, is [2:nj-1,2:nk-1]
par.dzeta=1/(par.nk-3);
par.dx=par.L/(par.ni-3);
par.dy=par.L/(par.nj-3);

par.zeta =-par.mu+([1:par.nk]-2)*par.dzeta;
par.x=([1:par.ni]-2)*par.dx;
par.y=([1:par.nj]-2)*par.dy;

%% spherical coordinates:
par.theta_north=10;
par.dphi=360/(par.ni-3);
par.dtheta=(90-par.theta_north)*2/(par.nj-3);
par.dphi_rad=par.dphi*pi/180;
par.dtheta_rad=par.dtheta*pi/180;
par.phi=([1:par.ni]-2)*par.dphi;
par.theta=par.theta_north+([1:par.nj]-2)*par.dtheta;
par.s=sind(par.theta);
par.c=cosd(par.theta);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h,R,Ta,To,Ts,qa,Hcr,S]=FIS_read_restart(par,var1,var2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B=NaN;

% Restart Filename Associated with previous EBM Run
restartfolder_EBM = '/u/scratch/f/fspauldi/exosnowballs/EBMRestart';

restart_filename = sprintf('%s/restart-EBM-expnum-%.2d-Q-%.2d-resnum-%.2d-eps-%.2d-%s.mat',restartfolder_EBM,par.EBM_expnum,par.Qo,par.N,100*var1,var2);

if exist(restart_filename,'file')
  fprintf(1,'initializing FIS @ N = %.2d with restart from the last EBM run.\n',par.N);
  B=ones(1,par.nj)*1e16*par.h0/par.R; % xx 
  load(restart_filename);

  % Set new fields to unusual defaults to catch bugs
  h(:,3) = NaN;
  R(:,3) = NaN;

else
  fprintf(1,'FIS error: no EBM restart file %s.\n',restart_filename);
  keyboard;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A=A_g(T)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Temperature dependence of ice viscosity, Goodman & Pierhumbert 2003.

if T<263.15
  A0=3.61e-13;
  Q=60e3;
else
  A0=1.734e3;
  Q=139e3;
end
R=8.3144621; % http://en.wikipedia.org/wiki/Gas_constant
A=A0*exp(-Q/(R*T));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [B,B_tN,B_tS,d_ftheta_v_N,d_ftheta_v_S]=calc_eff_viscosity_1d_sphere(n,par,T_surface,h,v,nn);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% effective viscosity for 1d model in spherical coordinates
% nn is the exponent in Glen's Law
% T_surface is the surface air temp
% par.T_f is the freezing temp
% A_g is a function that calculates the temp dependence of ice viscosity, A(T)

% Comments
% There needs to be a line that says what happens if retreating ice ever equals par.Nedge = 1 or par.nj - XX

B=NaN(par.nj,1);

B_tN = NaN;
B_tS = NaN;
d_ftheta_v_N=NaN;
d_ftheta_v_S=NaN;
% BC on B - what if edge reaches pole?  
B(1) = 0;
B(end) = 0;

dot_eps0=1e-14;
 

for i = 1:length(par.domain)
  j=par.domain(i);

  %% see ice_viscosity.m under AA folder in FISEBM/
  %% for justification that 30 levels is best approx.

  T=linspace(T_surface(j),par.T_f,30); 
  AA=T*NaN; % from air to ice-water interface temp
  
  for k=1:length(T)
    AA(k)=(A_g(T(k)))^(-1/3); 
  end

  % Solve for B at Nedge
  if j==par.Nedge

    %if n == 1
      %fprintf(1,'checkpoint B_tN\n');
      %keyboard;
    %end

    % Evaluate B_Tilde (@ j = par.Nedge)
    eps_xx = (1/(par.R*par.s(j)))*(v(j)*par.c(j));
    eps_yy = (1/par.R)*(v(j)-v(j-1))/(par.dtheta_rad);
    eps_zz = -(eps_xx + eps_yy); 
    eps_xy = 0;
    dot_eps=sqrt((eps_xx^2+eps_yy^2+eps_zz^2+2*eps_xy^2)/2);
    B_tN = (h(j)/par.R)*mean(AA)*(dot_eps+dot_eps0)^(1/nn-1);
     
    % Evaluate B_N (@ j = par.Nedge)
    d_ftheta_v_N = (1/4)*par.g*par.rho_i*h(j)^(2)*(1-par.mu)*B_tN^(-1) - (1/2)*v(j)*par.c(j)*par.s(j)^(-1);
    eps_xx = (1/(par.R*par.s(j)))*(v(j)*par.c(j));
    eps_yy = (1/par.R)*d_ftheta_v_N;
    eps_zz = -(eps_xx+eps_yy);
    eps_xy = 0;
    dot_eps=sqrt((eps_xx^2+eps_yy^2+eps_zz^2+2*eps_xy^2)/2);
    B(j)=(h(j)/par.R)*mean(AA)*(dot_eps+dot_eps0)^(1/nn-1);

  % Solve for B at Sedge
  elseif j==par.Sedge

    %if n == 1
      %fprintf(1,'checkpoint B_sN\n');
      %keyboard;
    %end

    % Evaluate B_tS (@ j = par.Sedge)
    eps_xx = (1/(par.R*par.s(j)))*(v(j)*par.c(j));
    eps_yy = (1/par.R)*(v(j+1)-v(j))/(par.dtheta_rad);
    eps_zz = -(eps_xx + eps_yy); 
    eps_xy = 0;
    dot_eps=sqrt((eps_xx^2+eps_yy^2+eps_zz^2+2*eps_xy^2)/2);
    B_tS = (h(j)/par.R)*mean(AA)*(dot_eps+dot_eps0)^(1/nn-1);
     
    % Evaluate B_tS (@ j = par.Sedge)
    d_ftheta_v_S = (1/4)*par.g*par.rho_i*h(j)^(2)*(1-par.mu)*B_tS^(-1) - (1/2)*v(j)*par.c(j)*par.s(j)^(-1);
    eps_xx = (1/(par.R*par.s(j)))*(v(j)*par.c(j));
    eps_yy = (1/par.R)*d_ftheta_v_S;
    eps_zz = -(eps_xx+eps_yy);
    eps_xy = 0;
    dot_eps=sqrt((eps_xx^2+eps_yy^2+eps_zz^2+2*eps_xy^2)/2);
    B(j)=(h(j)/par.R)*mean(AA)*(dot_eps+dot_eps0)^(1/nn-1);
  
  % Solve in Ice Shelf Interior
  else
    eps_xx=(1/(par.R*par.s(j)))*( ...
                               +v(j)*par.c(j));
    eps_yy=(1/par.R)*(v(j+1)-v(j-1))/(2*par.dtheta_rad);
    eps_zz=-(eps_xx+eps_yy);
    eps_xy=0;
    dot_eps=sqrt((eps_xx^2+eps_yy^2+eps_zz^2+2*eps_xy^2)/2);
    B(j)=(h(j)/par.R)*mean(AA)*(dot_eps+dot_eps0)^(1/nn-1);  
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = TDMAsolver(a,b,c,d)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% from http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
%a, b, c are the column vectors for the compressed tridiagonal matrix, d is the right vector
n = length(b); % n is the number of rows
 
% Modify the first-row coefficients
c(1) = c(1) / b(1);    % Division by zero risk.
d(1) = d(1) / b(1);    % Division by zero would imply a singular matrix.
 
for i = 2:n
    id = 1 / (b(i) - c(i-1) * a(i));  % Division by zero risk.
    c(i) = c(i)* id;                % Last value calculated is redundant.
    d(i) = (d(i) - d(i-1) * a(i)) * id;
end
 
% Now back substitute.
x(n) = d(n);
for i = n-1:-1:1
    x(i) = d(i) - c(i) * x(i + 1);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v,par]=solve_v_1d_matrix_form_sphere(n,par,B,B_tN,B_tS,h)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% solve diagnostic momentum equation for 1d linearized model in
%% spherical coordinates, by writing it in matrix form.

%% initialize variables:
v=NaN(par.nj,1);
a=NaN(par.nj,1);
b=a; c=a; d=a;

%% setup matrix and rhs:

for i = 2:(length(par.domain)-1)
  j = par.domain(i);

  if j == par.Nedge
      % No calc of B_jphalf
      % No calc of s_jphalf
      % BC of c(j) = 0
      % New a(j), b(j), d(j)
      B_jphalf = NaN;
      B_jmhalf =(B(j)+B(j-1))/2;
      %% sine at half locations:
      s_jphalf = NaN;
      s_jmhalf =(par.s(j)+par.s(j-1))/2;
      %% v-equations:
      a(j) = par.s(j)^(-1)*B_jmhalf*s_jmhalf/par.dtheta_rad^2 + B_jmhalf*s_jmhalf^(-1)*par.s(j-1)/par.dtheta_rad^2;
      
      b(j) = B(j)*par.s(j)^(-1)*par.s(j+1)/par.dtheta_rad^2 - par.c(j)*par.s(j)^(-1)*B(j)/(2*par.dtheta_rad) -par.c(j)*par.s(j)^(-1)*B(j)*par.s(j)^(-1)*par.s(j+1)/(2*par.dtheta_rad) - B(j)/par.dtheta_rad^2 - B_jmhalf*s_jmhalf^(-1)*par.s(j)/par.dtheta_rad^2 - par.s(j)^(-1)*B_jmhalf*s_jmhalf/par.dtheta_rad^2 - par.c(j)^(2)*par.s(j)^(-2)*B(j);

      c(j) = 0; % BC at terminus b/c v_Nedge+1 not defined
      test = 1; % check d(j) - artificially scaling back pressure - XX
      d(j) = par.g*par.rho_i*(1-par.mu)*h(j)*(h(j)-h(j-1))/(par.dtheta_rad) - par.g*par.rho_i*(1-par.mu*(test))*(h(j)^2)*B(j)*B_tN^(-1)*(1+par.s(j)^(-1)*par.s(j+1))/(4*par.dtheta_rad);

  elseif j == par.Sedge
      % No calc of B_jmhalf
      % No calc of s_jmhalf
      % BC of a(j) = 0
      % New a(j), b(j), d(j)
      B_jphalf = (B(j)+B(j+1))/2;
      B_jmhalf = NaN;
      %% sine at half locations:
      s_jphalf = (par.s(j)+par.s(j+1))/2;
      s_jmhalf = NaN;
      %% v-equations:
      a(j) = 0; % BC at terminus, v_Sedge-1 not defined 
     
      b(j) = -par.s(j)^(-1)*B_jphalf*s_jphalf/par.dtheta_rad^2 - B_jphalf*s_jphalf^(-1)*par.s(j)/par.dtheta_rad^2 - B(j)/par.dtheta_rad^2 - par.s(j)^(-2)*par.c(j)^(2)*B(j) + B(j)*par.s(j)^(-1)*par.s(j-1)/par.dtheta_rad^2 + B(j)*par.c(j)*par.s(j)^(-1)/(2*par.dtheta_rad) + B(j)*par.c(j)*par.s(j)^(-1)*par.s(j)^(-1)*par.s(j-1)/(2*par.dtheta_rad);

      c(j) = par.s(j)^(-1)*B_jphalf*s_jphalf/par.dtheta_rad^2 + B_jphalf*s_jphalf^(-1)*par.s(j+1)/par.dtheta_rad^2;
      test = 1; % check d(j) - artificially scaling back pressure - XX
      d(j) = par.g*par.rho_i*(1-par.mu)*h(j)*(h(j+1)-h(j))/(par.dtheta_rad) + par.g*par.rho_i*(1-par.mu*(test))*(h(j)^2)*B(j)*B_tS^(-1)*(1+par.s(j)^(-1)*par.s(j-1))/(4*par.dtheta_rad);
  else
    % Solve Ice Shelf Velocities in Both Hemispheres
    B_jphalf=(B(j)+B(j+1))/2;
    B_jmhalf=(B(j)+B(j-1))/2;
    %% sine at half locations:
    s_jphalf=(par.s(j)+par.s(j+1))/2;
    s_jmhalf=(par.s(j)+par.s(j-1))/2;
    %% v-equations:
    a(j)=par.s(j)^(-1)*B_jmhalf*s_jmhalf/par.dtheta_rad^2 ...
   +B_jmhalf*s_jmhalf^(-1)*par.s(j-1)/par.dtheta_rad^2;
    b(j)=...
      -par.s(j)^(-1)*B_jphalf*s_jphalf/par.dtheta_rad^2 ...
      -par.s(j)^(-1)*B_jmhalf*s_jmhalf/par.dtheta_rad^2 ...
      -B_jphalf*s_jphalf^(-1)*par.s(j)/par.dtheta_rad^2 ...
      -B_jmhalf*s_jmhalf^(-1)*par.s(j)/par.dtheta_rad^2 ...
      -(par.c(j)/par.s(j))*B(j)*par.s(j)^(-1)*par.c(j) ...
      ;
    
    c(j)=par.s(j)^(-1)*B_jphalf*s_jphalf/par.dtheta_rad^2 ...
   +B_jphalf*s_jphalf^(-1)*par.s(j+1)/par.dtheta_rad^2;
    d(j)=par.g*par.rho_i*(1-par.mu)*h(j)*(h(j+1)-h(j-1))/(2*par.dtheta_rad);
  end
end
test = 1; % XX
if par.Nedge~=0 
  par.aN = a(par.Nedge);
  par.bN = b(par.Nedge);
  par.cN = c(par.Nedge); % c=0
  par.dN = d(par.Nedge);
  par.d1N = par.g*par.rho_i*(1-par.mu)*h(par.Nedge)*(h(par.Nedge)-h(par.Nedge-1))/(par.dtheta_rad);
  par.d2N = par.g*par.rho_i*(1-par.mu*(test))*(h(par.Nedge)^2)*B(par.Nedge)*B_tN^(-1)*(1+par.s(par.Nedge)^(-1)*par.s(par.Nedge+1))/(4*par.dtheta_rad);
end
if par.Sedge~=0 
  par.aS = a(par.Sedge); % a=0 
  par.bS = b(par.Sedge);
  par.cS = c(par.Sedge);
  par.dS = d(par.Sedge);
  par.d1S = par.g*par.rho_i*(1-par.mu)*h(par.Sedge)*(h(par.Sedge+1)-h(par.Sedge))/(par.dtheta_rad);
  par.d2S = par.g*par.rho_i*(1-par.mu*(test))*h(par.Sedge)^(2)*B(par.Sedge)*B_tS^(-1)*(1+par.s(par.Sedge)^(-1)*par.s(par.Sedge-1))/(4*par.dtheta_rad);
end

% Inspect a, b, c, d
%keyboard;
%% prepare arrays for tridiagnonal solver:
%% note that v(1)=v(2)=v(par.nj-1)=v(par.nj-2)=0 and do not need to be solved for.
%% j=1 is north pole (~8 deg colat)!

if ((par.Nedge==0) && (par.Sedge==0))
  % Solve Using Eli's Code
  A1=zeros(par.nj-4,1); B1=A1; C1=A1; D1=A1; 
  A1(1)=0;
  B1(1)=b(3);
  C1(1)=c(3);
  D1(1)=d(3);
  for j=4:par.nj-3
    A1(j-2)=a(j);
    B1(j-2)=b(j);
    C1(j-2)=c(j);
    D1(j-2)=d(j);
  end
  A1(par.nj-4)=a(par.nj-2);
  B1(par.nj-4)=b(par.nj-2);
  C1(par.nj-4)=0;
  D1(par.nj-4)=d(par.nj-2);

  %% call tridiagonal solver:
  x = TDMAsolver(A1,B1,C1,D1);
  v(3:par.nj-2)=x;

  %% north and south b.c.:
  v(1)=0;
  v(2)=0;
  v(par.nj-1)=0;
  v(par.nj)=0;

elseif par.Sedge==0
  % Partial Ice NH, Total Ice SH
  A1N = zeros(par.Nedge-2,1); B1N=A1N; C1N=A1N; D1N=A1N; 
  A1N(1)=0;
  B1N(1)=b(3);
  C1N(1)=c(3);
  D1N(1)=d(3);
  for j=4:par.Nedge
    A1N(j-2)=a(j);
    B1N(j-2)=b(j);
    C1N(j-2)=c(j);
    D1N(j-2)=d(j);
  end
  %% call tridiagonal solver for NH
  x = TDMAsolver(A1N,B1N,C1N,D1N);
  v(3:par.Nedge)=x;

  A1S = zeros(par.EQ-2,1); B1S=A1S; C1S=A1S; D1S=A1S; 
  for j=par.EQ:par.nj-3
    % j - par.EQ + 1 ~ 1: 
    A1S(j-par.EQ+1)=a(j);
    B1S(j-par.EQ+1)=b(j);
    C1S(j-par.EQ+1)=c(j);
    D1S(j-par.EQ+1)=d(j);
  end
  A1S(end)=a(par.nj-2);
  B1S(end)=b(par.nj-2);
  C1S(end)=0;
  D1S(end)=d(par.nj-2);
  %% call tridiagonal solver for SH
  x = TDMAsolver(A1S,B1S,C1S,D1S);
  v(par.EQ:par.nj-2)=x;

  % from array definition, v(par.Nedge+1:par.EQ-1) = NaN

  %% north pole b.c.:
  v(1)=0;
  v(2)=0;
  v(par.nj-1)=0;
  v(par.nj)=0;

elseif par.Nedge==0
  % Total Ice NH, Partial Ice SH
  A1N=zeros(par.EQ-2,1); B1N=A1N; C1N=A1N; D1N=A1N; 
  A1N(1)=0;
  B1N(1)=b(3);
  C1N(1)=c(3);
  D1N(1)=d(3);
  for j=4:par.EQ
    A1N(j-2)=a(j);
    B1N(j-2)=b(j);
    C1N(j-2)=c(j);
    D1N(j-2)=d(j);
  end
  %% call tridiagonal solver for the NH:
  x = TDMAsolver(A1N,B1N,C1N,D1N);
  v(3:par.EQ)=x;

  A1S=zeros(par.nj-par.Sedge-1,1); B1S=A1S; C1S=A1S; D1S=A1S; 
  for j=par.Sedge:par.nj-3
    A1S(j-par.Sedge+1)=a(j);
    B1S(j-par.Sedge+1)=b(j);
    C1S(j-par.Sedge+1)=c(j);
    D1S(j-par.Sedge+1)=d(j);
  end
  A1S(end)=a(par.nj-2);
  B1S(end)=b(par.nj-2);
  C1S(end)=0;
  D1S(end)=d(par.nj-2);

  %% call tridiagonal solver for the SH:
  x = TDMAsolver(A1S,B1S,C1S,D1S);
  v(par.Sedge:par.nj-2)=x;

  % from array definition, v(par.EQ+1:par.Sedge-1) = NaN
  %% north and south b.c.:
  v(1)=0;
  v(2)=0;
  v(par.nj-1)=0;
  v(par.nj)=0;

elseif ((par.Nedge~=0)&&(par.Sedge~=0))

  % Partial Ice NH, Partial Ice SH
  A1N = zeros(par.Nedge-2,1); B1N=A1N; C1N=A1N; D1N=A1N; 
  A1N(1)=0;
  B1N(1)=b(3);
  C1N(1)=c(3);
  D1N(1)=d(3);
  for j=4:par.Nedge
    A1N(j-2)=a(j);
    B1N(j-2)=b(j);
    C1N(j-2)=c(j);
    D1N(j-2)=d(j);
  end
  %% call tridiagonal solver for NH
  x = TDMAsolver(A1N,B1N,C1N,D1N);
  v(3:par.Nedge)=x;

  A1S=zeros(par.nj-par.Sedge-1,1); B1S=A1S; C1S=A1S; D1S=A1S; 
  for j=par.Sedge:par.nj-3
    A1S(j-par.Sedge+1)=a(j);
    B1S(j-par.Sedge+1)=b(j);
    C1S(j-par.Sedge+1)=c(j);
    D1S(j-par.Sedge+1)=d(j);
  end
  A1S(end)=a(par.nj-2);
  B1S(end)=b(par.nj-2);
  C1S(end)=0;
  D1S(end)=d(par.nj-2);

  %% call tridiagonal solver for the SH:
  x = TDMAsolver(A1S,B1S,C1S,D1S);
  v(par.Sedge:par.nj-2)=x;

  %% north and south b.c.:
  v(1)=0;
  v(2)=0;
  v(par.nj-1)=0;
  v(par.nj)=0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function R = find_R(h,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% finds the partial ice coverage field R of a height array, h or delta h
R = NaN(length(h),1);
for j = 1:length(h)
  if h(j)>=par.Hcr
    % total ice cover
    R(j) = 1;
  elseif ((0<=h(j)) && (h(j)<par.Hcr))
    % partial or zero ice cover
    R(j) = h(j)/par.Hcr;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h,R,par] = ogrid(ogi,h,R,v_np1,par,S)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(ogi)
  j = ogi(i);

    % ogi ~ ocean grid index: for terminuses with no adjacent subgrid, ogi gives the
    % index of the adjacent ocean grid cell
    % Note that 'negative' advection from terminus is negated in delta_h
  if j < par.EQ
    delta_hN = par.dt*h(par.Nedge,2)*v_np1(par.Nedge)/(par.R*par.dtheta_rad);
    if delta_hN < 0
      delta_hN=0;
    end
    delta_h = par.dt*S(j) + delta_hN;
 
    if delta_h > 0
      % Advance of Ice Shelf into Next Ocean Grid Cell
      if delta_h >= par.Hcr
        % grid scale advection
        h(j,3) = delta_h;
        R(j,3) = 1;
      elseif delta_h < par.Hcr
        % subgrid scale advection
        h(j,3) = 0;
        R(j,3) = find_R(delta_h,par);
      end         
    elseif delta_h <= 0
      % No change to ocean grid cell ~ no ice advected from terminus
      h(j,3) = 0;
      R(j,3) = 0;
    end

  elseif j > par.EQ 
    delta_hS=-par.dt*h(par.Sedge,2)*v_np1(par.Sedge)/(par.R*par.dtheta_rad);
    if delta_hS < 0
      delta_hS = 0;
    end
    delta_h = par.dt*S(j) + delta_hS;

    if delta_h > 0
      % Advance of Ice Shelf into Next Ocean Grid Cell
      if delta_h >= par.Hcr
        % grid scale advection
        h(j,3) = delta_h;
        R(j,3) = 1;
      elseif delta_h < par.Hcr
        % subgrid scale advection
        h(j,3) = 0;
        R(j,3) = find_R(delta_h,par);
      end
    elseif delta_h <= 0
      % No change to ocean grid cell ~ no ice advected from terminus
      h(j,3) = 0;
      R(j,3) = 0;
    end

  elseif j == par.EQ
    if ((par.Nedge==(par.EQ-1)) && (par.Sedge==(par.EQ+1)))
      delta_hN = par.dt*h(par.Nedge,2)*v_np1(par.Nedge)/(par.R*par.dtheta_rad);
      delta_hS = -par.dt*h(par.Sedge,2)*v_np1(par.Sedge)/(par.R*par.dtheta_rad);
      if delta_hN < 0
        delta_hN = 0;
      end
      if delta_hS < 0
        delta_hS = 0;
      end       
      delta_h = par.dt*S(j) + delta_hN + delta_hS;           

      if delta_h > 0
        if delta_h >= par.Hcr
          % grid scale advection
          h(j,3) = delta_h;
          R(j,3) = 1;
          % equatorial gap closed
        elseif delta_h < par.Hcr
          % subgrid scale advection
          h(j,3) = 0;
          R(j,3) = find_R(delta_h,par);
        end
      elseif delta_h <= 0
        % No change to ocean grid cell
        h(j,3) = 0;
        R(j,3) = 0;
      end
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h,R,par] = sgrid(sgi,h,R,v_np1,par,S,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Subgrid Parametrization: Advance and Retreat %%
% Note: at any given time step, any additional "melt" beyond the subgrid cell will be
% categorized as heat input into the ocean

for i = 1:length(sgi)
  j = sgi(i);
  if j < par.EQ
    delta_hN = par.dt*h(par.Nedge,2)*v_np1(par.Nedge)/(par.R*par.dtheta_rad);
    if delta_hN < 0
      delta_hN = 0;
    end
    delta_h = par.dt*S(j) + delta_hN;    
     % Since 0 < R(j,2) < 1, the volume of subgrid j is
    h_sg = R(j,2)*par.Hcr;
    if delta_h <= 0
      % Retreat of Ice Shelf
      if abs(delta_h) > h_sg
        % subgrid cell j melts entirely
        % additional melt categorized as heat input into the ocean 
        h(j,3) = 0; % subgrid melts
        R(j,3) = 0; % subgrid melts
        % apply remainder melt to ocean:

      elseif abs(delta_h) < h_sg
        % subgrid cell j only partially melts
        h(j,3) = 0;
        h_sg = h_sg - abs(delta_h);  % find new subgrid height
        R(j,3) = find_R(h_sg,par); % update R in subgrid cell j
      end
    elseif delta_h > 0
      % Advance of Ice Shelf
      h_sg_2R1 = par.Hcr - h_sg;

      % If advance into next cell, identify the cell type of j+1
      if R(j+1,2)==0
        % ocean cell
        h_adv_2fill_jp1 = par.Hcr;
      elseif R(j+1,2)==1
        % grid cell (isolated shelves)
        h_adv_2fill_jp1 = 0;
        %fprintf(1,'sg=%2d advancing into gc=%2d\n',j,j+1);
 
        % do we build up ice in j? 
      elseif (R(j+1,2)>0) & (R(j+1,2)<1) 
        % subgrid cell
        h_adv_2fill_jp1 = par.Hcr - R(j+1,2)*par.Hcr;
        % additional ice mass goes where?
      end

      if delta_h > (h_sg_2R1 + h_adv_2fill_jp1) 
        % subgrid cells j & j+1 fill to par.Hcr (limits on advance)
        R(j,3) = 1; % subgrid fills/converts to grid
        h(j,3) = par.Hcr; % subgrid fills/converts to grid
        R(j+1,3) = 1; % next ocean subgrid fills/converts to grid
        h(j+1,3) = par.Hcr; % next ocean subgrid fills/converts to grid
      elseif delta_h > h_sg_2R1
        % subgrid cell j fills entirely and partly fills next subgrid cell j+1
        R(j,3) = 1; % subgrid fills/converts to grid
        h(j,3) = par.Hcr; % subgrid fills/converts to grid   
        % find adv height of ice to subgrid cell j+1
        delta_h = delta_h - h_sg_2R1;
        % update R value in subgrid cell j+1`
        R(j+1,3) = R(j+1,2)+find_R(delta_h,par);
      elseif delta_h < h_sg_2R1
        % subgrid cell j partially fills
        h(j,3) = 0;
        h_sg = h_sg + delta_h; % partially fill subgrid  
        R(j,3) = find_R(h_sg,par); % update subgrid field
      end
    end
  elseif j > par.EQ
    delta_hS = -par.dt*h(par.Sedge,2)*v_np1(par.Sedge)/(par.R*par.dtheta_rad);
    if delta_hS < 0
      delta_hS = 0;
    end
    delta_h = par.dt*S(j) + delta_hS;
    % Since 0 < R(j,2) < 1, the volume of subgrid j is
    h_sg = R(j,2)*par.Hcr;
    if delta_h <= 0
      % Retreat of Ice Shelf
      if abs(delta_h) > h_sg
        % subgrid cell j melts entirely
        % additional melt categorized as heat input into the ocean
        h(j,3) = 0; % subgrid melts
        R(j,3) = 0; % subgrid melts
        % apply remainder melt to ocean:
 
        % par.Sedge remains unchanged
      elseif abs(delta_h) < h_sg
        % subgrid cell j only partially melts
        h(j,3) = 0;
        h_sg = h_sg - abs(delta_h);  % find new subgrid height
        R(j,3) = find_R(h_sg,par); % update R in subgrid cell j
      end
    elseif delta_h > 0
      % Advance of Ice Shelf
      h_sg_2R1 = par.Hcr - h_sg;

      % If advance into next cell, identify the cell type of j-1
      if R(j-1,2)==0
        % ocean cell
        h_adv_2fill_jp1 = par.Hcr;
      elseif R(j-1,2)==1
        % grid cell (isolated shelves)
        h_adv_2fill_jp1 = 0;
        %fprintf(1,'sg=%2d advancing into gc=%2d\n',j,j-1);
       
        % do we build up ice in j? 
      elseif (R(j-1,2)>0) & (R(j-1,2)<1) 
        % subgrid cell
        h_adv_2fill_jp1 = par.Hcr - R(j-1,2)*par.Hcr;
        % additional ice mass goes where?
      end

      if delta_h > (h_sg_2R1 + h_adv_2fill_jp1)
        % subgrid cells j & j+1 fill to par.Hcr (limits on advance)
        R(j,3) = 1; % subgrid fills/converts to grid
        h(j,3) = par.Hcr; % subgrid fills/converts to grid
        R(j-1,3) = 1; % next ocean subgrid fills/converts to grid
        h(j-1,3) = par.Hcr; % next ocean subgrid fills/converts to grid
      elseif delta_h > h_sg_2R1
        % subgrid cell j fills entirely and partly fills next subgrid cell j-1
        R(j,3) = 1; % subgrid fills/converts to grid
        h(j,3) = par.Hcr; % subgrid fills/converts to grid   
        % find adv height of ice in ocean subgrid cell j-1
        delta_h = delta_h - h_sg_2R1;
        % update value in ocean subgrid cell j+1
        R(j-1,3) = R(j-1,2)+find_R(delta_h,par);
      elseif delta_h < h_sg_2R1
        % subgrid cell j partially fills
        h(j,3) = 0;
        h_sg = h_sg + delta_h; % partially fill subgrid  
        R(j,3) = find_R(h_sg,par); % update subgrid field 
      end
    end
  elseif j==par.EQ
%% Note: symmetric melt/advance might be problematic 
    if ((par.Nedge==(par.EQ-1)) && (par.Sedge==(par.EQ+1)))
      %% equatorial mass balance involves both hemispheres     
      delta_hN = par.dt*h(par.Nedge,2)*v_np1(par.Nedge)/(par.R*par.dtheta_rad);
      delta_hS = -par.dt*h(par.Sedge,2)*v_np1(par.Sedge)/(par.R*par.dtheta_rad);
      if delta_hN < 0 
        delta_hN = 0;
      end
      if delta_hS < 0
        delta_hS = 0;
      end
      delta_h = par.dt*S(j) + delta_hN + delta_hS;
      % Since 0 < R(j,2) < 1, the volume of subgrid j is
      h_sg = R(j,2)*par.Hcr;

      if delta_h <= 0
        %% Retreat of Ice Shelf
        if abs(delta_h) > h_sg 
          % subgrid cell j melts entirely and 
          % additional melt categorized as heat input into the ocean
          h(j,3) = 0; % subgrid melts
          R(j,3) = 0; % subgrid melts
          % apply remainder melt to ocean:

          % unchanged terminuses
        elseif abs(delta_h) < h_sg
          % subgrid cell j only partially melts
          h(j,3) = 0;
          h_sg = h_sg - abs(delta_h); % find new subgrid height
          R(j,3) = find_R(h_sg,par); % update R in subgrid cell j
        end
      elseif delta_h > 0
        % Advance of Ice Shelf in regime of subgrid @ equator
        % symmetric terminuses @ EQ-1 on either side
        h_sg_2R1 = par.Hcr - h_sg;
        if delta_h > h_sg_2R1 
          % subgrid fills entirely, but growth limited to Hcr
          h(j,3) = par.Hcr; % subgrid fills/converts to grid
          R(j,3) = 1; % subgrid fills/converts to grid
          % transition from partial to global
        elseif delta_h < h_sg_2R1
          % subgrid cell j partially fills
          h(j,3) = 0;
          h_sg = h_sg + delta_h; % partially fill subgrid
          R(j,3) = find_R(h_sg,par); % update subgrid field
        end 
      end
      
    elseif ((par.Nedge==(par.EQ-1)) && (par.Sedge~=(par.EQ+1)))
      %% equatorial mass balance involves NH
      if par.Sedge==0
        fprintf(2,'***error: par.Sedge cannot equal zero\n');
        keyboard;
      end
      delta_hN = par.dt*h(par.Nedge,2)*v_np1(par.Nedge)/(par.R*par.dtheta_rad);
      if delta_hN < 0 
        delta_hN = 0;
      end
      delta_h = par.dt*S(j) + delta_hN;
      % Since 0 < R(j,2) < 1, the volume of subgrid j is
      h_sg = R(j,2)*par.Hcr;

      if delta_h <= 0
        % Retreat of Ice Shelf
        if abs(delta_h) > h_sg
          % subgrid cell j melts entirely and
          % additional melt cateogrized as heat input into the ocean
          h(j,3) = 0; % subgrid melts
          R(j,3) = 0; % subgrid melts
          % apply remainder melt to ocean:
 
         % par.Nedge remains unchanged
        elseif abs(delta_h) < h_sg
          % subgrid cell j only partially melts
          h(j,3) = 0;
          h_sg = h_sg - abs(delta_h);  % find new subgrid height
          R(j,3) = find_R(h_sg,par); % update R in subgrid cell j
        end
      elseif delta_h > 0
        % Equatorial Advance of Ice Shelf from NH in subgrid scale
        h_sg_2R1 = par.Hcr - h_sg;
        if delta_h > h_sg_2R1
          % subgrid fills entirely, but growth limited to Hcr
          h(j,3) = par.Hcr; % subgrid fills/converts to grid
          R(j,3) = 1; % subgrid fills/converts to grid
          % transition from partial to global hemisphere
        elseif delta_h < h_sg_2R1
          % subgrid cell j partially fills
          h(j,3) = 0;
          h_sg = h_sg + delta_h; % partially fill subgrid
          R(j,3) = find_R(h_sg,par); % update subgrid field
        end
      end      
    elseif((par.Nedge~=(par.EQ-1)) && (par.Sedge==(par.EQ+1)))
      %% equatorial mass balance involves SH
      if par.Nedge==0
        fprintf(2,'***error:par.Nedge cannot equal zero');
        keyboard;
      end
      delta_hS = -par.dt*h(par.Sedge,2)*v_np1(par.Sedge)/(par.R*par.dtheta_rad);
      if delta_hS < 0
        delta_hS = 0;
      end
      delta_h = par.dt*S(j) + delta_hS;
      h_sg = R(j,2)*par.Hcr;

      if delta_h <= 0
        % Retreat of Ice Shelf
        if abs(delta_h) > h_sg
          % subgrid cell j melts entirely and
          % additional melt categorized as heat input into the ocean
          h(j,3) = 0; % subgrid melts
          R(j,3) = 0; % subgrid melts
          % apply remainder melt to the ocean:
          
          % par.Sedge remains unchanged
        elseif abs(delta_h) < h_sg
          % subgrid cell j only partially melts
          h(j,3) = 0;
          h_sg = h_sg - abs(delta_h);  % find new subgrid height
          R(j,3) = find_R(h_sg,par); % update R in subgrid cell j
        end
      elseif delta_h > 0
        % Equatorial Advance of Ice Shelf from SH in subgrid scale
        h_sg_2R1 = par.Hcr - h_sg;
        if delta_h > h_sg_2R1
          % subgrid fills entirely, but growth limited to Hcr
          h(j,3) = par.Hcr; % subgrid fills/converts to grid
          R(j,3) = 1; % subgrid fills/converts to grid
          % transition from partial to global hemisphere
        elseif delta_h < h_sg_2R1
          % subgrid cell j partially fills
          h(j,3) = 0;
          h_sg = h_sg + delta_h; % partially fill subgrid
          R(j,3) = find_R(h_sg,par); % update subgrid field
        end
      end     
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h,R] = exsgrid(h,par,R,S,exsgi,v_np1,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(exsgi)
  j = exsgi(i); 
  if isnan(R(j,3))==1
    % This code runs when sgi advance hasn't updated a R(eSC,3)
    if j <= par.EQ
      delta_h = par.dt*S(j);
      h_sg = R(j,2)*par.Hcr;

      if delta_h <= 0 
	% Retreat of Ice Shelf
	if abs(delta_h) > h_sg
	  % subgrid cell j melts entirely
	  % additional melt categorized as heat input into the ocean
	  h(j,3) = 0;
	  R(j,3) = 0;
	  % apply remainder melt to ocean

	  % par.Nedge remains unchange
	elseif abs(delta_h) < h_sg
	  % subgrid cell j only partially melts
          h(j,3) = 0;
	  h_sg = h_sg - abs(delta_h); % find new subgrid height
	  R(j,3) = find_R(h_sg,par); % update R in subgrid cell j
	end
      elseif delta_h > 0
	% Advance of Ice Shelf
	h_sg_2R1 = par.Hcr - h_sg;
	if delta_h > h_sg_2R1
	  % subgrid cells j fill entirely
	  h(j,3) = par.Hcr;
	  R(j,3) = 1;
	  %fprintf(1,'warning: extraneous cell full - check terminus');
	  %keyboard;
	  % additional ice mass goes where? 
	elseif delta_h < h_sg_2R1
	  % subgrid cell j partly fills
          h(j,3) = 0;
	  h_sg = h_sg + delta_h; % partly fill subgrid
	  R(j,3) = find_R(h_sg,par); % update R at j
	end  
      end
    elseif j > par.EQ
      delta_h = par.dt*S(j);
      h_sg = R(j,2)*par.Hcr;

      if delta_h <= 0
	% Retreat of Ice Shelf
	if abs(delta_h) > h_sg
	  % subgrid cell j melts entirely
	  % additional melt categorized as heat input into the ocean
	  h(j,3) = 0;
	  R(j,3) = 0;
	  % apply remainder melt to ocean:

	elseif abs(delta_h) < h_sg
	  % subgrid cell j only partially melts
          h(j,3) = 0;
	  h_sg = h_sg - abs(delta_h); % find new subgrid height 
	  R(j,3) = find_R(h_sg,par); % update R in subgrid cell j
	end    
      elseif delta_h > 0
	% Advance of Ice Shelf
	h_sg_2R1 = par.Hcr - h_sg;
	if delta_h > h_sg_2R1
	  % subgrid cell j fills entirely
	  h(j,3) = par.Hcr;
	  R(j,3) = 1;          
	  %fprintf(1,'warning: extraneous cell full - check terminus');
	  %keyboard;
	  % additional ice mass goes where?
	elseif delta_h < h_sg_2R1
	  % subgrid cell j partially fills
          h(j,3) = 0;
	  h_sg = h_sg + delta_h; % partially fill subgrid
	  R(j,3) = find_R(h_sg,par); % update R at j
	end
      end
    end
  elseif isnan(R(j,3))==0 
    % This code runs when sgi advance has updated a R(eSC,3)
    % The sgrid function has already applied advection mass balance to 
    % the eSC cell and updated it in R(j,3). Thus, h_sg uses 
    % (and consequently updates) R(j,3), instead of R(j,2). 
    % The source function is then applied to the cell. 
    if j <= par.EQ
      delta_h = par.dt*S(j);

      if R(j,3)==1
        h_sg = h(j,3);
      else
        h_sg = R(j,3)*par.Hcr;
      end

      if delta_h <= 0 
	% Retreat of Ice Shelf
	if abs(delta_h) > h_sg
	  % subgrid cell j melts entirely
	  % additional melt categorized as heat input into the ocean
	  h(j,3) = 0;
	  R(j,3) = 0;
	  % apply remainder melt to ocean

	  % par.Nedge remains unchange
	elseif abs(delta_h) < h_sg
	  % subgrid cell j only partially melts
	  h_sg = h_sg - abs(delta_h); % find new subgrid height
          if h_sg >= par.Hcr
            h(j,3) = h_sg; 
          else 
            h(j,3) = 0;
          end
	  R(j,3) = find_R(h_sg,par); % update R in subgrid cell j
	end
      elseif delta_h > 0
	% Advance of Ice Shelf
        h(j,3) = h_sg + delta_h;
        if h(j,3)>=par.Hcr
          % h stays the same
          R(j,3) = find_R(h(j,3),par);
        else
          % subgrid
          R(j,3) = find_R(h(j,3),par);
          h(j,3) = 0;
        end
      end
    elseif j > par.EQ
      delta_h = par.dt*S(j);

      if R(j,3)==1
        h_sg = h(j,3);
      else
        h_sg = R(j,3)*par.Hcr;
      end

      if delta_h <= 0
	% Retreat of Ice Shelf
	if abs(delta_h) > h_sg
	  % subgrid cell j melts entirely
	  % additional melt categorized as heat input into the ocean
	  h(j,3) = 0;
	  R(j,3) = 0;
	  % apply remainder melt to ocean:

	elseif abs(delta_h) < h_sg
	  % subgrid cell j only partially melts
	  h_sg = h_sg - abs(delta_h); % find new subgrid height 
          if h_sg >= par.Hcr
            h(j,3) = h_sg; 
          else 
            h(j,3) = 0;
          end
	  R(j,3) = find_R(h_sg,par); % update R in subgrid cell j
	end    
      elseif delta_h > 0
	% Advance of Ice Shelf
        h(j,3) = h_sg + delta_h;
        if h(j,3)>=par.Hcr
          % h stays the same
          R(j,3) = find_R(h(j,3),par);
        else
          % subgrid
          R(j,3) = find_R(h(j,3),par);
          h(j,3) = 0;
        end
      end
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h,R] = isogrid(h,par,R,S,isogci,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mass Balance in Isolated Ice Shelves (grid cells interior to a subgrid cell)
% Grid to Subgrid Transitions carried out in main 'integrate' loop

for i = 1:length(isogci)
  j = isogci(i);

  delta_h = par.dt*S(j);
  if delta_h > 0
    % Ice Shelf Growth
    h(j,3) = h(j,2)+delta_h;
    R(j,3) = find_R(h(j,3),par);
  elseif delta_h <= 0
    % Ice Shelf Shrinkage
    h(j,3) = h(j,2) - abs(delta_h);

    if h(j,3)<0
      h(j,3) = 0;
      R(j,3) = 0;
    elseif h(j,3)<par.Hcr && h(j,3)>0
      R(j,3) = find_R(h(j,3),par);
      h(j,3) = 0;
    else
      R(j,3) = find_R(h(j,3),par);
      % h(j,3) as calculated above
    end

  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h,R] = OOgrid(h,par,R,S,oogci)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mass Balance in Open Ocean Grid Cells
% Defined as an open ocean cell not directly adjacent to a shelf terminus
% Grid to Subgrid Transitions carried out in main 'integrate' loop
for i = 1:length(oogci)
  j = oogci(i);

  delta_h = par.dt*S(j);

  if delta_h > 0
    % Ice Growth on Ocean? XX
    if delta_h>par.Hcr
      h(j,3) = delta_h;
      R(j,3) = find_R(h(j,3),par);
    else
      R(j,3) = find_R(delta_h,par);
      h(j,3) = 0;
    end
  elseif delta_h <= 0
    % additional heat input to ocean? XX
    h(j,3) = 0;
    R(j,3) = 0;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Ts = Ts_extrap_FIS(h,R,Ts,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function extrapolates surface ice temperatures when ice
% cover advances to previously ice-free regions.
% It also returns the surface temperature field to NaN if 
% ice-free regions are generated. 

% Retreat to Open Ocean from Ice Cover
all_Ts_rt = find(R(:,2)==0 & R(:,1)>0);
if length(all_Ts_rt)>0
  for i = 1:length(all_Ts_rt)
    j = all_Ts_rt(i);
    Ts(j) = NaN;
  end
end

% General case of ice expanding to new latitudes 
% i.e. advance from open ocean to ice cover

% Ts Extrapolation Function - 04/20/2020
newice = find(R(:,2)>0 & R(:,1)==0);

if length(newice)>0
  for i = 1:length(newice)
    j = newice(i);
    if j<par.EQ
      % Closest Ice GC Poleward of j in NH
      j_NH = max(find(R(1:j-1,2)>0 & R(1:j-1,1)>0)); 
      Ts(j) = Ts(j_NH);
    elseif j>par.EQ
      % Closest Ice GC Poleward of j in SH
      j_SH = min(find(R((j+1):end,2)>0 & ...
                     R((j+1):end,1)>0))...
                     + j;
      Ts(j) = Ts(j_SH);
    else
      % Closest Ice GC Poleward of EQ
      j_NH = max(find(R(1:j-1,2)>0 & R(1:j-1,1)>0)); 
      j_SH = min(find(R((j+1):end,2)>0 & ...
                     R((j+1):end,1)>0))...
                     + j;
      cNH = abs(j_NH-j);
      cSH = abs(j_SH-j);
      if min(cNH,cSH)==cNH
        % Ice cell in NH is closer
        Ts(j) = Ts(j_NH) 
      elseif min(cNH,cSH)==cSH 
        % Ice cell in SH is closer
        Ts(j) = Ts(j_SH) 
      else
        % Same distance in either hemisphere
        Ts(j) = Ts(j_NH) % = Ts(j_SH)  
      end
    end
  end
end

% Simplest Option: Simply set new ice to Ts=Tf
%if length(all_Ts_adv)>0
  %Ts(all_Ts_adv) = par.T_f;
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Tex = Ts_extrap_EBM(j,R,Ts,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a helper function to To_check in the EBM. 
% FSA 04/20/2020
% Input Fields: j, R(:,:), Ts(:,3)

if j<par.EQ
    if j==1
	% Boundary Condition
	Tex = par.T_f;
    else
        % Closest Ice GC Poleward of j in NH
        j_NH = max(find(R(1:j-1,3)>0 & R(1:j-1,2)>0)); 
        Tex = Ts(j_NH);
    end
elseif j>par.EQ
    if j==par.nj
	% Boundary Condition
	Tex = par.T_f;
    else
        % Closest Ice GC Poleward of j in SH
        j_SH = min(find(R((j+1):end,3)>0 & ...
                    R((j+1):end,2)>0))...
                    + j;
        Tex  = Ts(j_SH);
    end
else
    % Closest Ice GC Poleward of EQ
    j_NH = max(find(R(1:j-1,3)>0 & R(1:j-1,2)>0)); 
    j_SH = min(find(R((j+1):end,3)>0 & ...
                    R((j+1):end,2)>0))...
                    + j;
    cNH = abs(j_NH-j);
    cSH = abs(j_SH-j);
    if min(cNH,cSH)==cNH
        % Ice cell in NH is closer
        Tex = Ts(j_NH) 
    elseif min(cNH,cSH)==cSH 
        % Ice cell in SH is closer
        Tex = Ts(j_SH) 
    else
        % Same distance in either hemisphere
        Tex = Ts(j_NH) % = Ts(j_SH)  
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function To = To_extrap_old(h,R,To,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function extrapolates mixed layer ocean temperatures when 
% open-ocean regions are generated.
% It also resets the ocean temperature to freezing when open ocean
% regions become ice covered.

%all_To_gen = find(R==0 & R_initial>0);
all_To_gen = find(R(:,2)==0 & R(:,1)>0);

if length(all_To_gen)>0
  for i = 1:length(all_To_gen)
    j = all_To_gen(i);
    if j <= par.EQ
      minval = j+1;
      maxval = j+3;
      indexrange = all_To_gen<=maxval & all_To_gen>=minval;
      if length(indexrange)~=0
        % select closest ocean cell in vicinity to extrapolate To
        j_extrap = min(indexrange);
        if j_extrap>=1 && j_extrap<=par.EQ
          To(j) = To(j_extrap);
        else
          To(j) = par.T_f;
        end
      else
        % default to freezing temp 
        To(j) = par.T_f;
      end
    elseif j > par.EQ
      minval = j-3;
      maxval = j-1;
      indexrange = all_To_gen<=maxval & all_To_gen>=minval;
      if length(indexrange)~=0
        % select closest ocean cell in vicinity to extrapolate To
        j_extrap = max(indexrange);
        if j_extrap<=par.nj & j_extrap>par.EQ
          To(j) = To(j_extrap);
        else
          To(j) = par.T_f;
        end
      else
        % default to freezing temp 
        To(j) = par.T_f;
      end
    end
  end
end

% Initial open ocean regions that become ice covered
%all_To_elim = find(R>0 & R_initial==0);
%all_To_elim = find(R(:,2)>0 & R(:,1)==0);
%if length(all_To_elim)>0
  %for i = 1:length(all_To_elim)
    %j = all_To_elim(i);
    %To(j) = par.T_f;
  %end
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Fi = FIS_F_i(h,par,R,v)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% longterm horizontal convergences of ice at each grid cell - xx
dh = zeros(par.nj,1);
Fi = zeros(par.nj,1);

v(find(isnan(v)==1))=0;

for j = 2:(par.nj-1)
  dv_dtheta = (v(j+1)*par.s(j+1)*h(j+1) - v(j-1)*par.s(j-1)*h(j-1))/(2*par.dtheta_rad);
  Fi(j) = par.rho_i*(par.R*par.s(j))^(-1)*dv_dtheta;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [par] = FIS_iceline(h,R,par,text)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function determines the ice line and ends FIS if:
% (0) no ice at initialization of FIS
% (1) ice retreats to the poles in FIS
% (2) ice advances to the equator in FIS
% (3) ice line stabilizes 
% (4) the global ice area changes more than the area
%     corresponding to an equatorial 4 deg lat strip
%     (enough to significantly feed back on climate)
%% Operation includes 
% i) final round of plots
% ii) make restart file
% iii) FIS early termination

par.text = text;

%% Nedge/Sedge Update %%
if strcmp('start',text)==1
  t = 2;
elseif strcmp('end',text)==1
  t = 3;
end

arg_NH = min(find(h(1:par.EQ,t)==0));
if length(arg_NH)~=0
  if arg_NH==1
    Nedge=NaN;
  elseif arg_NH>=2
    Nedge=arg_NH-1;
  else
    fprintf(1,'error: edge\n');
    keyboard;
  end
elseif sum(h(1:par.EQ,t)>0)==ceil(par.nj/2)
  % no h=0 found - ice covered
  Nedge=0;
else
  fprintf(1,'unforseen Nedge routine\n');
  keyboard;
end

arg_SH = max(find(h(par.EQ:end,t)==0));
if length(arg_SH)~=0
  if arg_SH==par.nj
    Sedge=NaN;
  elseif arg_SH<=(par.nj-1)
    Sedge = arg_SH+45;
  else
    fprintf(1,'error: edge\n');
    keyboard;
  end
elseif sum(h(par.EQ:end,t)>0)==ceil(par.nj/2)
  % no h=0 found - ice covered
  Sedge = 0;
else
  fprintf(1,'unforseen Nedge routine\n');
  keyboard;
end

% Operation (0)
if (isnan(Nedge)==1 || isnan(Sedge)==1) & t==3
  par.noice = 1; 
  fprintf(1,sprintf('FISpar.noice = 1, EBM_expnum %.2d, n = %.2d\n',par.EBM_expnum,par.n));
elseif (isnan(Nedge)==1 || isnan(Sedge)==1) & t==2
  par.noice = 1; 
  fprintf(1,sprintf('FISpar.noice = 1, EBM_expnum %.2d, n = %.2d\n',par.EBM_expnum,0));
end

% Operation (1)
if (Nedge<=2 & Nedge>=1) | (Sedge>=(par.nj-1) & Sedge<=par.nj)
  if t==2
    par.icelatpole = 1;
    fprintf(1,sprintf('FISpar.icelatpole = 1, EBM_expnum %.2d, n = %.2d\n',par.EBM_expnum,1));
  elseif t==3
    par.icelatpole = 1;
    fprintf(1,sprintf('FISpar.icelatpole = 1, EBM_expnum %.2d, n = %.2d\n',par.EBM_expnum,par.n));
  end
end

% Operation (2) - XX
%if isnan(Nedge)==1 & isnan(Sedge)==1
  %fprintf(1,sprintf('FISpar.icelateq = 1, EBM_expnum %.2d, n = %.2d\n',par.EBM_expnum,n));
  %par.icelateq = 1;
%end

% Operation (3)
%if par.n == par.nt
  %if Nedge==par.Nedge_initial & Sedge==par.Sedge_initial
    %par.icelatstable = 1;
    %fprintf(1,sprintf('FISpar.icelatstable = 1, EBM_expnum %.2d, FIS_n = %.2d\n',par.EBM_expnum,par.n));
    % final round of plots
    % make restart file
    % flag run for partial equilibrium
    % update equilibrium plot  
  %end
%end

%%% Operation (4) %%%

if strcmp('end',text)==1

  % If global land ice area changes more than area of 4 degrees latitude strip
  xi = sum(par.R_initial);
  xf = sum(R(:,2));
  
  if abs(xi-xf)>=2
    par.icelatlim = 1;
    fprintf(1,sprintf('Operation 4 Early FIS Termination: FISpar.icelatlim = 1, EBM_expnum %.2d, FIS_n = %.2d\n',par.EBM_expnum,par.n));
    % flag run for early termination
  end
end

% Test Operation 4 before implementation - not working correctly
%if strcmp('end',text)==1
  %xi = par.h_initial;
  %xf = h(:,2);
  %xi(xi>0)=1;
  %xf(xf>0)=1;

  %xi_N = xi(1:par.EQ);
  %xi_S = xi(par.EQ:end);
  %xf_N = xf(1:par.EQ);
  %xf_S = xf(par.EQ:end);

  %z_N = sum(xi_N~=xf_N);
  %z_S = sum(xi_S~=xf_S);

  %if abs(sum(xi_N)-sum(xf_N))>=2 | abs(sum(xi_S)-sum(xf_S))>=2 ...
     %| z_N>=2 | z_S>=2
    %par.icelatlim = 1;
    %fprintf(1,sprintf('Operation 4 Early FIS Termination: FISpar.icelatlim = 1, EBM_expnum %.2d, FIS_n = %.2d\n',par.EBM_expnum,par.n));
    % flag run for early termination
  %end
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Sn,dhdt_cond] = update_S(dhdt_odiff,h,R,S,To,Ts,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This subroutine ensures snowfall occurs over ice and rain over ocean,
% in the source function submitted to the ice flow model. 
% Conductive Flux
k_i = 10^4; % heat conductivity of ice (m^2/s) 
dhdt_cond = Cforc(h,R,To,Ts,par);

Sn = zeros(par.nj,1);
index = find(isnan(Ts)==0);
Sn(index) = S(index);
Sn = Sn + dhdt_cond + dhdt_odiff;

% Boundary Conditions on Source Term
%Sn(1) = 0;
%Sn(end) = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function par = integrate_h_1d_sphere(par,var1,var2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ice flow in 1 horizontal dimension using spherical coordinates.

% Create Folder to Export FIS Plots
plotfolder_FIS = '/u/scratch/f/fspauldi/exosnowballs/FISFigures';
if ~exist(sprintf('%s',plotfolder_FIS),'file')
  [status,msg] = mkdir(sprintf('%s',plotfolder_FIS));
end

% Create Folder to Export FIS Restarts
restartfolder_FIS = '/u/scratch/f/fspauldi/exosnowballs/FISRestart';
if ~exist(sprintf('%s',restartfolder_FIS),'file')
  [status,msg] = mkdir(sprintf('%s',restartfolder_FIS));
end

% Specify part of filename
fig_filename_1d='1d-sphere-nonlinear';

plotfolder = sprintf('%s/FIS-exp-%.2d-Q-%.2d-%s-N-%.2d-eps-%.2d-%s/',plotfolder_FIS,par.EBM_expnum,par.Qo,fig_filename_1d,par.N,100*var1,var2);
%if par.N==500
if ~exist(sprintf('%s/FIS-exp-%.2d-Q-%.2d-%s-N-%.2d-eps-%.2d-%s',plotfolder_FIS,par.EBM_expnum,par.Qo,fig_filename_1d,par.N,100*var1,var2),'file') & (par.N==1 || floor(par.N/par.Nplot)*par.Nplot==par.N || par.N==par.Nt) 
    [status,msg] = mkdir(sprintf('%s/FIS-exp-%.2d-Q-%.2d-%s-N-%.2d-eps-%.2d-%s',plotfolder_FIS,par.EBM_expnum,par.Qo,fig_filename_1d,par.N,100*var1,var2));
elseif exist(sprintf('%s/FIS-exp-%.2d-Q-%.2d-%s-N-%.2d-eps-%.2d-%s',plotfolder_FIS,par.EBM_expnum,par.Qo,fig_filename_1d,par.N,100*var1,var2),'file') & (par.N==1 || floor(par.N/par.Nplot)*par.Nplot==par.N || par.N==par.Nt) 
    [status,msg] = rmdir(sprintf('%s/FIS-exp-%.2d-Q-%.2d-%s-N-%.2d-eps-%.2d-%s',plotfolder_FIS,par.EBM_expnum,par.Qo,fig_filename_1d,par.N,100*var1,var2),'s');
    [status,msg] = mkdir(sprintf('%s/FIS-exp-%.2d-Q-%.2d-%s-N-%.2d-eps-%.2d-%s',plotfolder_FIS,par.EBM_expnum,par.Qo,fig_filename_1d,par.N,100*var1,var2));
end
%end

fprintf(1,'running FIS @ N = %.2d, EBM_expnum %.2d, Qo %.2d, eps %.2d, ice %s\n',par.N,par.EBM_expnum,par.Qo,100*var1,var2);

if 0
  %% print diagnostics regarding the time step size:
  fprintf(1,'(par.R*par.dtheta_rad)^2/kappa=%g, dx/(U=50m/yr)=%g, dt=%g (yrs)\n' ...
          ,((par.R*par.dtheta_rad)^2/par.kappa_2d)/par.year ...
          ,((par.R*par.dtheta_rad)/(50/par.year))/par.year ...
          ,par.dt/par.year);
end


%% exponent of Glen's law.   
nn=3;

%% initialize variables:
div_hv_n=NaN(par.nj,1);
div_hv_np1=NaN(par.nj,1);
kappa_del2_h_n=NaN(par.nj,1);
kappa_del2_h_np1=NaN(par.nj,1);
v_n=zeros(par.nj,1);
v_np1=zeros(par.nj,1); % vn plus 1

mask=ones(par.nj,1); 
nan_mask=mask;
mask(1)=0; mask(par.nj)=0;
nan_mask(mask==0)=NaN;

% Track h, R, and v over time (FSA 2023)
htrack = NaN(par.nj,par.nt);
Rtrack = NaN(par.nj,par.nt);
vtrack = NaN(par.nj,par.nt);

% Create Folder to Export h, R, and v over time (FSA 2023)
trackingfolder_FIS = '/u/scratch/f/fspauldi/exosnowballs/tracking';
if ~exist(sprintf('%s',trackingfolder_FIS),'file')
  [status,msg] = mkdir(sprintf('%s',trackingfolder_FIS));
end

% Load Model Variables
[h,R,Ta,To,Ts,qa,Hcr,S_init]=FIS_read_restart(par,var1,var2);
par.Hcr = Hcr;

% Boundary Condition on Source Function
%S_init(1) = 0;
%S_init(par.nj) = 0;
S = S_init;

% Sensible Heat from Ocean to Ice Base in Source Function
diff_To = zeros(par.nj,1);
%index = find(isnan(Ts(2:(par.nj-1),2))==0) + 1;
index = 2:(par.nj-1);
for i = 1:length(index)
    j = index(i);
  s_jphalf = 0.5*(par.s(j)+par.s(j+1)); 
  s_jmhalf = 0.5*(par.s(j)+par.s(j-1));
  diff_To(j) = (par.rho_o*par.c_o*par.h_o*par.D_o/(par.R^2*par.s(j)))*(1/par.dtheta_rad^2)*(s_jphalf*(To(j+1,2)-To(j,2)) - s_jmhalf*(To(j,2)-To(j-1,2)));
end

dhdt_odiff = zeros(par.nj,1); % Sensible Heat is obv. zero when h=0
dhdt_geo = -par.G/(par.rho_i*par.L_f); 
dhdt_odiff(index) = -par.Beta*(To(index,2) - par.T_f)/(par.rho_i*par.L_f) + dhdt_geo;

% Assign initial Ts
T_surface = Ts(:,1:2);
T_ocean = To(:,1:2);

% Define Nedge_initial
arg_NH = min(find(h(1:par.EQ,2)==0));
if length(arg_NH)~=0
  if arg_NH==1
    par.Nedge_initial=NaN;
  elseif arg_NH>=2
    par.Nedge_initial=arg_NH-1;
  else
    fprintf(1,'error: edge\n');
    keyboard;
  end
elseif sum(h(1:par.EQ,2)>0)==ceil(par.nj/2)
  % no h=0 found - ice covered
  par.Nedge_initial=0;
else
  fprintf(1,'unforseen Nedge routine\n');
  keyboard;
end

% Define Sedge_initial
arg_SH = max(find(h(par.EQ:end,2)==0));
if length(arg_SH)~=0
  if arg_SH==par.nj
    par.Sedge_initial=NaN;
  elseif arg_SH<=(par.nj-1)
    par.Sedge_initial = arg_SH+45;
  else
    fprintf(1,'error: edge\n');
    keyboard;
  end
elseif sum(h(par.EQ:end,2)>0)==ceil(par.nj/2)
  % no h=0 found - ice covered
  par.Sedge_initial = 0;
else
  fprintf(1,'unforseen Nedge routine\n');
  keyboard;
end

par.h_initial = h(:,2);
par.R_initial = R(:,2);
par.To_initial = To(:,2);
par.Ta_initial = Ta(:,2);
par.To_init_ave = globmean(To(:,2),par);
par.Ta_init_ave = globmean(Ta(:,2),par);

% Checking Input File for Open Ocean
par = FIS_iceline(h,R,par,'start');

if par.noice==1 || par.icelatpole==1
  return;
end

print = 1;

%% Time step h-equation:
for n=1:par.nt  %%%%%%%%%%%%%%%%%%%%%%%%%% for n = 1 or n = par.nt/2 

  if n==print
    FIS_n = n
    if n>1
      print = print + 100;
    else
      print = print + 99;
    end
  end

  %fprintf(1,sprintf('global mean albedo = %.2d\n',globmean(h(:,2),par))); % xx

  % Update time step
  par.n=n;
  time_kyr=n*par.dt/par.year/1000;
  
  % Update v_n
  v_n = v_np1;

  % Update subgrid indices (sgi)
  % Locate (if any) new subgrid indices (existing subgrid cells in R)
  new_sg=find((h(:,2)<par.Hcr) & (h(:,2)>0));
  if length(new_sg)~=0
    R(new_sg,2) = find_R(h(new_sg,2),par);
    for i = 1:length(new_sg)
      j = new_sg(i);
      h(j,2) = 0;
    end
  end

  % Update par.Nedge
  arg_NH = min(find(h(1:par.EQ,2)==0));
  if length(arg_NH)~=0
    if arg_NH==1
      par.Nedge=NaN;
    elseif arg_NH>=2
      par.Nedge=arg_NH-1;
    else
      fprintf(1,'error: edge\n');
      keyboard;
    end
  elseif sum(h(1:par.EQ,2)>0)==ceil(par.nj/2)
    % no h=0 found - ice covered
    par.Nedge=0;
  else
    fprintf(1,'unforseen Nedge routine\n');
    keyboard;
  end

  % Update par.Sedge
  arg_SH = max(find(h(par.EQ:end,2)==0));
  if length(arg_SH)~=0
    if arg_SH==par.nj
      par.Sedge=NaN;
    elseif arg_SH<=(par.nj-1)
      par.Sedge = arg_SH+45;
    else
      fprintf(1,'error: edge\n');
      keyboard;
    end
  elseif sum(h(par.EQ:end,2)>0)==ceil(par.nj/2)
    % no h=0 found - ice covered
    par.Sedge = 0;
  else
    fprintf(1,'unforseen Nedge routine\n');
    keyboard;
  end

  % Check for invalid par.Nedge
  if (((par.Nedge<=2) | (par.Nedge>=par.EQ)) & (par.Nedge~=0))
   fprintf(1,'par.Nedge = %.2d\n',par.Nedge);
   fprintf(1,'*** par.Nedge invalid: advise stopping model\n');
   keyboard;
  end 
  % Check for invalid par.Sedge
  if (((par.Sedge<=par.EQ) | (par.Sedge>=(par.nj-1))) & (par.Sedge~=0))
   fprintf(1,'par.Sedge = %.2d\n',par.Sedge);
   fprintf(1,'*** par.Sedge invalid: advise stopping model\n');
   keyboard;
  end 
 
  % Identify subgrid indices (sgi) and extra subgrid indices (exsgi)
  all_sgi = find(R(:,2)>0 & R(:,2)<1);
  if length(all_sgi)==0
    sgilen = 0;
    exsgilen = 0;
  else
    if par.Nedge==0 & par.Sedge==0
      sgilen=0;
      exsgi = all_sgi;
      exsgilen = length(exsgi);
      fprintf(1,'this an error?\n');
      keyboard;
    elseif par.Nedge==0
      sgi = all_sgi(find(all_sgi==(par.Sedge-1)));
      sgilen = length(sgi);
      if sgilen==0
	exsgi = all_sgi;
	exsgilen = length(exsgi);
      else
	exsgi = all_sgi(1:(end-1));
	exsgilen = length(exsgi);
      end
    elseif par.Sedge==0
      sgi = all_sgi(find(all_sgi==(par.Nedge+1)));
      sgilen = length(sgi);
      if sgilen==0
	exsgi = all_sgi;
	exsgilen = length(exsgi);
      else
	exsgi = all_sgi(2:end);
	exsgilen = length(exsgi);
      end
    else
      sgi = all_sgi(find(all_sgi==(par.Nedge+1) | all_sgi==(par.Sedge-1)));
      sgilen = length(sgi);

      if sgilen==0
	exsgi = all_sgi;
	exsgilen = length(exsgi);
      elseif sgilen==1
	if sgi>par.Nedge & sgi<=par.EQ
	  exsgi = all_sgi(2:end);
	  exsgilen = length(exsgi);
	elseif sgi<par.Sedge & sgi>=par.EQ
	  exsgi = all_sgi(1:(end-1));
	  exsgilen = length(exsgi);
	else
	  fprintf(1,'error: sgi calc - two hemis\n');
	  keyboard;
	end
      elseif sgilen==2
	exsgi = all_sgi(2:(end-1));
	exsgilen = length(exsgi);
      end
    end
    v_np1(:) = 0; % XX
  end

  %% Check new sgi 
  if sgilen==2
    if (sgi(:) < par.EQ)
      fprintf(2,'error: multiple SCs (not eSCs) in one hemisphere');
      keyboard;
    end
    if (sgi(:) > par.EQ)
      fprintf(2,'error: multiple SCs (not eSCs) in one hemisphere');
      keyboard;
    end
  end

  % Check for invalid NH subgrid-terminus regime
  if ((par.Nedge==0) & (min(all_sgi)<=par.EQ)) 
   fprintf(1,'par.Nedge = %.2d; min(sgi)=%.2d\n',par.Nedge,min(sgi));
   fprintf(1,'*** invalid NH subgrid-terminus regime: advise stopping model\n');
  end
  % Check for invalid SH subgrid-terminus regime
  if ((par.Sedge==0) & (max(all_sgi)>=par.EQ)) 
   fprintf(1,'par.Sedge = %.2d; max(sgi)=%.2d\n',par.Sedge,max(sgi));
   fprintf(1,'*** invalid SH subgrid-terminus regime: advise stopping model\n');
  end

  %% Edited isogci and oogci to work when there aren't two terminuses
  % Identify isolated grid cells (isogci)
  all_gci = find(R(:,2)==1);
  if par.Nedge==0 && par.Sedge~=0
    minval = par.EQ+1;
    maxval = par.Sedge-1;
    if minval > maxval
      isogcilen = 0;
    else
      indexrange = all_gci > minval & all_gci < maxval;
      isogci = all_gci(indexrange);
      isogcilen = length(isogci);
    end
  elseif par.Sedge==0 && par.Nedge~=0
    minval = par.Nedge+1;
    maxval = par.EQ-1;
    if minval > maxval
      isogcilen = 0;
    else
      indexrange = all_gci > minval & all_gci < maxval;
      isogci = all_gci(indexrange);
      isogcilen = length(isogci);
    end
  else
    minval = par.Nedge+1;
    maxval = par.Sedge-1;
    indexrange = all_gci > minval & all_gci < maxval;
    isogci = all_gci(indexrange);
    isogcilen = length(isogci);
  end


  % Identify Open Ocean Grid Cells (oogci)
  all_ogi = find(R(:,2)==0);
  if length(all_ogi)>0
    if par.Nedge==0 & par.Sedge~=0
      minval = par.EQ+1;
      maxval = par.Sedge-1;
      if minval > maxval
        oogcilen = 0;
      else
        indexrange = all_ogi > minval & all_ogi < maxval;
        oogci = all_ogi(indexrange);
        oogcilen = length(oogci);
      end
    elseif par.Nedge~=0 & par.Sedge==0
      minval = par.Nedge+1;
      maxval = par.EQ-1;
      if minval > maxval
        oogcilen = 0;
      else
        indexrange = all_ogi > minval & all_ogi < maxval;
        oogci = all_ogi(indexrange);
        oogcilen = length(oogci);
      end
    else
      minval = par.Nedge+1;
      maxval = par.Sedge-1;
      indexrange = all_ogi > minval & all_ogi < maxval;
      oogci = all_ogi(indexrange);
      oogcilen = length(oogci);
    end
  else
    oogcilen = 0;
  end

  % Update the operable domain
  if ((par.Nedge==0)&(par.Sedge==0))
    par.domain = [2:par.nj-1];
  elseif ((par.Nedge~=0)&(par.Sedge==0))
    par.domain = [2:par.Nedge,ceil(par.nj/2):par.nj-1];
  elseif ((par.Nedge==0)&(par.Sedge~=0))
    par.domain = [2:ceil(par.nj/2),par.Sedge:par.nj-1];
  elseif ((par.Nedge~=0)&(par.Sedge~=0))
    par.domain = [2:par.Nedge,par.Sedge:par.nj-1];
  end

  % Extrapolate Surface Temperature of the Ice, if needed
  T_surface(:,2) = Ts_extrap_FIS(h(:,2),R,T_surface(:,2),par);

  % Extrapolate Mixed Layer Ocean Temperature, if needed
  %T_ocean(:,2) = To_extrap(h(:,2),R,T_ocean(:,2),par);

  % Transition from Partial to Global requires default v_np1 field
  if (par.Nedge==0 & par.Sedge==0) & sum(isnan(v_n))>0
    v_np1(:) = 0;
  end

  % Update Source Function
  [S,dhdt_cond] = update_S(dhdt_odiff,h(:,2),R(:,2),S_init,T_ocean(:,2),T_surface(:,2),par);

  if n==1;
    % this requires same h,R initial conditions for n=1:2 
    % in other words, the initial conditions should be equilibrium fields
    v_n=v_np1; 
    B = ones(par.nj,1);
  end


  % Edit v-np1 & B if advance of ice produces new grid cell for V Loop
  % This step is important because initialization v-np1 is the 
  % same as v-n, which has a smaller real-valued domain.
  % This step is only an issue if v_def or v_undef are <3 or >par.nj-2

  % New subgrids have a B, but no v
  v_undef = find(R(:,2)>0 & R(:,2)<1 & ...
          (isnan(B)==1 | isnan(v_np1))==0);
  if length(v_undef)>0
    minval = 3;
    maxval = par.nj-2;
    indexrange = v_undef>=minval & v_undef<=maxval;
    v_undef = v_undef(indexrange);
    for j = 1:length(v_undef)
      v_np1(v_undef(j)) = NaN; % uninitalize v in new sg
      B(v_undef(j)) = 1; % initialize B in new sg
    end
  end

  % New grids have a B and a v 
  v_def = find(h(:,2)>0 & ...
          (isnan(B)==1 | isnan(v_np1==1)));
  if length(v_def)>0
    minval = 3;
    maxval = par.nj-2;
    indexrange = v_def>=minval & v_def<=maxval;
    v_def = v_def(indexrange);
    for j = 1:length(v_def)
      v_np1(v_def(j)) = 0; % initalize v in new grid
      B(v_def(j)) = 1; % initalize B in new grid
    end
  end

  % Equating # of NaNs in v_n and v_np1 for Height Loop
  % Note: If v_n has more NaNs than v_np1, then the height loop below will
  % iterate over NaNs in v_n while iterating over real number values in v_np1.  
  if sum(isnan(v_n))>sum(isnan(v_np1))
    for j=3:(par.nj-2)
      if (isnan(v_n(j))==1) & (isnan(v_np1(j))==0)
        v_n(j) = v_np1(j);
      end
    end
  end


  %% calculate velocity:
  %% -------------------
  max_iter_eff_viscosity=1000;
  par.iter_eff_viscosity=0;
  diff_eff_viscosity=1;
  while par.iter_eff_viscosity<=max_iter_eff_viscosity ...
        && diff_eff_viscosity>1.e-4;
    %% Average viscosity over relevant temperature range:
    B_old=B;

    [B,B_tN,B_tS,d_ftheta_v_N,d_ftheta_v_S]=calc_eff_viscosity_1d_sphere(n,par,T_surface(:,2),h(:,2),v_np1,nn);

    % Flow vs No-Flow Switch
    if strcmp(var2,'flow')==1
      [v_np1,par]=solve_v_1d_matrix_form_sphere(n,par,B,B_tN,B_tS,squeeze(h(:,2)));
    elseif strcmp(var2,'no-flow')==1
      v_np1 = zeros(par.nj,1);
      vmask = B; vmask(find(isnan(vmask)==0))=1;
      v_np1 = v_np1.*vmask; 
    else
      fprintf(1,'error: flow condition unspecified\n');
      keyboard;
    end


    max_B=max(B(find(mask~=0)));
    diff_eff_viscosity=max(abs(B-B_old))/max_B;
    par.iter_eff_viscosity=par.iter_eff_viscosity+1;
    if par.iter_eff_viscosity==max_iter_eff_viscosity
      fprintf(1,'*** at n=%d, reached max_iter_eff_viscosity=%d at n=%d, with diff_eff_viscosity=%g\n' ...
              ,n,max_iter_eff_viscosity,par.n,diff_eff_viscosity);
    end
  end
  

  if (length(find(h(:,2)==0))~=length(find(isnan(v_np1)==1)))&(isogcilen==0)
    fprintf(2,'***error: # of h=0 ~= # of v_np1 NaNs\n');
    keyboard;
  end


  %% advance h in time using second order Adams Bashforth:
  %% NOTE: code currently assumes h does not go below a certain minimum


  %% Reinitialize variables:
  div_hv_n=NaN(par.nj,1);
  div_hv_np1=NaN(par.nj,1);
  kappa_del2_h_n=NaN(par.nj,1);
  kappa_del2_h_np1=NaN(par.nj,1);
  RHS_n = NaN(par.nj,1);
  RHS_np1 = NaN(par.nj,1);


  for i = 1:length(par.domain)
    j = par.domain(i);
    if j == par.Nedge 
      s_jmhalf=0.5*(par.s(j)+par.s(j-1));
      s_jphalf=0.5*(par.s(j)+par.s(j+1));
      kappa_del2_h_n(j)=par.kappa*( ...
        (1/(par.R^2*par.s(j)))* ...
        (-s_jmhalf*(h(j,1)-h(j-1,1))*mask(j-1)*mask(j-2) ...
         )/par.dtheta_rad^2);
    
      div_hv_n(j)= (1/(par.R*par.s(j)))*(...
        (par.s(j)*h(j,1)*v_n(j)-par.s(j-1)*h(j-1,1)*v_n(j-1)) ...
        /(par.dtheta_rad) ...
        );
      RHS_n=-div_hv_n(j)+kappa_del2_h_n(j)+S(j);
    
      kappa_del2_h_np1(j)=par.kappa*( ...
        (1/(par.R^2*par.s(j)))* ...
        (-s_jmhalf*(h(j,2)-h(j-1,2))*mask(j-1)*mask(j-2) ...
        )/par.dtheta_rad^2);
    
      div_hv_np1(j)= (1/(par.R*par.s(j)))*(...
        (par.s(j)*h(j,2)*v_np1(j)-par.s(j-1)*h(j-1,2)*v_np1(j-1)) ...
        /(par.dtheta_rad) ...
        );
      RHS_np1=-div_hv_np1(j)+kappa_del2_h_np1(j)+S(j);
      
      h(j,3)=h(j,2)+par.dt*(1.5*RHS_np1-0.5*RHS_n); 

  elseif j == par.Sedge

      s_jmhalf=0.5*(par.s(j)+par.s(j-1));
      s_jphalf=0.5*(par.s(j)+par.s(j+1));
      kappa_del2_h_n(j)=par.kappa*( ...
        (1/(par.R^2*par.s(j)))* ...
        (s_jphalf*(h(j+1,1)-h(j,1))*mask(j+2)*mask(j+1) ...
        )/par.dtheta_rad^2);
    
      div_hv_n(j)= (1/(par.R*par.s(j)))*(...
        (par.s(j+1)*h(j+1,1)*v_n(j+1)-par.s(j)*h(j,1)*v_n(j)) ...
        /(par.dtheta_rad) ...
        );
      RHS_n=-div_hv_n(j)+kappa_del2_h_n(j)+S(j);
    
      kappa_del2_h_np1(j)=par.kappa*( ...
        (1/(par.R^2*par.s(j)))* ...
        (s_jphalf*(h(j+1,2)-h(j,2))*mask(j+2)*mask(j+1) ...
        )/par.dtheta_rad^2);
    
      div_hv_np1(j)= (1/(par.R*par.s(j)))*(...
        (par.s(j+1)*h(j+1,2)*v_np1(j+1)-par.s(j)*h(j,2)*v_np1(j)) ...
        /(par.dtheta_rad) ...
        );
      RHS_np1=-div_hv_np1(j)+kappa_del2_h_np1(j)+S(j);
    
      h(j,3)=h(j,2)+par.dt*(1.5*RHS_np1-0.5*RHS_n); 

    else
      s_jmhalf=0.5*(par.s(j)+par.s(j-1));
      s_jphalf=0.5*(par.s(j)+par.s(j+1));
      kappa_del2_h_n(j)=par.kappa*( ...
        (1/(par.R^2*par.s(j)))* ...
        (s_jphalf*(h(j+1,1)-h(j,1))*mask(j)*mask(j+1) ...
        -s_jmhalf*(h(j,1)-h(j-1,1))*mask(j)*mask(j-1) ...
        )/par.dtheta_rad^2);
    
      div_hv_n(j)= (1/(par.R*par.s(j)))*(...
        (par.s(j+1)*h(j+1,1)*v_n(j+1)-par.s(j-1)*h(j-1,1)*v_n(j-1)) ...
        /(2*par.dtheta_rad) ...
        );
      RHS_n=-div_hv_n(j)+kappa_del2_h_n(j)+S(j);
    
      kappa_del2_h_np1(j)=par.kappa*( ...
        (1/(par.R^2*par.s(j)))* ...
        (s_jphalf*(h(j+1,2)-h(j,2))*mask(j)*mask(j+1) ...
        -s_jmhalf*(h(j,2)-h(j-1,2))*mask(j)*mask(j-1) ...
        )/par.dtheta_rad^2);
    
      div_hv_np1(j)= (1/(par.R*par.s(j)))*(...
        (par.s(j+1)*h(j+1,2)*v_np1(j+1)-par.s(j-1)*h(j-1,2)*v_np1(j-1)) ...
        /(2*par.dtheta_rad) ...
        );
      RHS_np1=-div_hv_np1(j)+kappa_del2_h_np1(j)+S(j);
    
      h(j,3)=h(j,2)+par.dt*(1.5*RHS_np1-0.5*RHS_n); 

    end

    % Final Assignment of h and R fields
    if h(j,3)<=0
      % Ice Melts to Open Ocean
      h(j,3) = 0;
      R(j,3) = 0;
      % xx - additional heat input to ocean?
    elseif h(j,3)<par.Hcr && h(j,3)>0
      R(j,3) = find_R(h(j,3),par);
      h(j,3) = 0;
    else
      % grid scale ice thickness change
      R(j,3) = find_R(h(j,3),par);
    end
  end

  %%% Grid and Subgrid Parametrization: Advance and Retreat %%%
  if (sgilen==0 && (par.Nedge==0 && par.Sedge==0))
    % global ice cover - no advection at ice edge
    % contintuity eq solved in applicable domain
    % R(:,3) = find_R(h(:,3),par);
    % WARNING: if you update R here then you erase new sgs!
  elseif ((sgilen==0) && ((par.Nedge~=0) | (par.Sedge~=0)))
    % partial ice cover, no subgrid
    if ((par.Nedge==(par.EQ-1)) && (par.Sedge==(par.EQ+1)))
      %fprintf(1,'checkpoint 1: g NH & SH into EQ\n');
      %keyboard;
      % advection into EQ ocean cell on grid scale
      ogi = par.EQ;
      [h,R,par] = ogrid(ogi,h,R,v_np1,par,S);
    elseif par.Nedge==0
      % NH ice shelf reaches EQ, advection in SH
      ogi = par.Sedge-1;
      [h,R,par] = ogrid(ogi,h,R,v_np1,par,S); 
      %fprintf(1,'checkpoint: NH @ EQ, g SH');
      %keyboard;
    elseif par.Sedge==0
      % SH ice shelf reaches EQ, advection in NH
      ogi = par.Nedge+1;
      [h,R,par] = ogrid(ogi,h,R,v_np1,par,S);
      %fprintf(1,'checkpoint: SH @ EQ, g NH');
      %keyboard;
    else
      % advection in NH & SH 
      ogi = [par.Nedge+1,par.Sedge-1];
      [h,R,par] = ogrid(ogi,h,R,v_np1,par,S); 
      %fprintf(1,'checkpoint: g in NH & SH\n');
      %keyboard;
    end
  elseif sgilen==1
    % partial ice cover, 1 subgrid cell
    if par.Nedge==0
      % NH ice shelf reaches EQ, subgrid advection in SH
      % check sgi = par.Sedge-1
      [h,R,par] = sgrid(sgi,h,R,v_np1,par,S,n);
      %fprintf(1,'NH @ EQ, sg SH\n');
      %keyboard;
    elseif par.Sedge==0 
      % SH ice shelf reaches EQ, subgrid in NH
      % check sgi = par.Nedge+1
      [h,R,par] = sgrid(sgi,h,R,v_np1,par,S,n);
      %fprintf(1,'checkpoint: SH @ EQ, sg NH\n');
      %keyboard;
    else
      if sgi < par.EQ
        % subgrid advection in NH, grid advection in SH
        ogi = par.Sedge-1;
        [h,R,par] = ogrid(ogi,h,R,v_np1,par,S);
        [h,R,par] = sgrid(sgi,h,R,v_np1,par,S,n);
        %fprintf(1,'checkpoint: sg NH, g SH\n');
        %keyboard;
      elseif sgi > par.EQ
        % subgrid advection in SH, grid advection in NH 
        ogi = par.Nedge+1;
        [h,R,par] = ogrid(ogi,h,R,v_np1,par,S);
        [h,R,par] = sgrid(sgi,h,R,v_np1,par,S,n);  
        %fprintf(1,'checkpoint: sg SH, g NH\n');
        %keyboard;
      elseif sgi == par.EQ 
        % subgrid advection into EQ from NH and SH
        % check N = EQ-1, S = EQ+1
        [h,R,par] = sgrid(sgi,h,R,v_np1,par,S,n);
        %fprintf(1,'checkpoint: sgi=par.EQ\n');
        %keyboard;
      end
    end
  elseif sgilen==2
    % partial ice cover, 2 subgrid cells
    [h,R,par] = sgrid(sgi,h,R,v_np1,par,S,n); 
    %fprintf(1,'checkpoint: sg NH & SH\n');
    %keyboard;
  end


  %% User's Note:  Mass Balance of Extra Subgrids is determined
  %% by the source function. If a hemisphere has more than two subgrids, the
  %% subgrids more than two or more cells away from the terminus are referred to
  %% as 'extra subgrids'. In the first step, the continuity equation is 
  %% solved in the subgrid cells that border each terminus. If the terminus isn't
  %% adjacent to a subgrid, then the ogrid function calculates the advection of 
  %% ice into the adjacent open ocean cell. 

  %%% Mass Balance of Extra Subgrids %%%
  if exist('exsgi','var') & exsgilen>0
    [h,R] = exsgrid(h,par,R,S,exsgi,v_np1,n);
  end

  %%% Mass Balance of Isolated Ice Shelves (Grid Cells) %%%
  if exist('isogci','var') & isogcilen>0
     [h,R] = isogrid(h,par,R,S,isogci,n);
  end

  %%% Mass Balance of Open Ocean Grid Cell %%%
  % Different class of ogi, which is the adjacent ocean cell to a terminus.
  % Open Ocean Grid Cell (oogci) defined as an ocean cell not adjacent to a terminus. 
  if exist('oogci','var') & oogcilen>0
    [h,R] = OOgrid(h,par,R,S,oogci);
  end


  if 0
    %% verify that total advection and diffusion vanish:
    sum_source=sum(S.*par.s)%/mean(abs(S.*par.s))
    sum_dif_n=sum(kappa_del2_h_n.*par.s')%/mean(abs(kappa_del2_h_n.*par.s'))
    sum_dif_np1=sum(kappa_del2_h_np1.*par.s')%/mean(abs(kappa_del2_h_np1.*par.s'))
    sum_adv_n=sum(div_hv_n.*par.s')%/mean(abs(div_hv_n.*par.s'))
    sum_adv_np1=sum(div_hv_np1.*par.s')%/mean(abs(div_hv_np1.*par.s'))
    pause
  end
  
  %% set north and south boundary conditions of h,R:
  h(1,3)=h(2,3);
  h(par.nj,3)=h(par.nj-1,3);
  R(1,3) = R(2,3);
  R(par.nj,3) = R(par.nj-1,3); 

  % Check for NaNs in h,R 
  if sum(isnan(h(:,3)))>0 | sum(isnan(R(:,3)))>0
    fprintf(1,'checkpoint why NaN\n');
    keyboard;
  end

  %% check for non-physical values of h & R:
  found_negative_h=0;
  if (min(h(:,3))<0 | (min(R(:,3))<0 | max(R(:,3))>1))
    fprintf(1,'min(h)=%g; max(h)=%g; min(R)=%g; max(R)=%g;',...
    min(h(:,3)),max(h(:,3)),min(R(:,3)),max(R(:,3)));
    fprintf(1,'*** stopping due to h<0 or R<0, n=%d ***\n',par.n);
    found_negative_h=1;
  end

  % Check for Unphysical Values of To and Freeze Ice if To<Tf
  %  or melt ice if To>Tf under Thick Ice
  %[h(:,3),R(:,3),T_ocean(:,2),T_surface(:,2),dhdt_frz]=To_check(h(:,3),n,par,R(:,3),T_ocean(:,2),T_surface(:,2));

  % Update par for Equilibrium Ice Line Checks
  par = FIS_iceline(h,R,par,'end');
  
  % Prepare Rplot, hplot
  Rplot = R(:,2);
  hplot = h(:,2);
  hplot(find(hplot==0)) = NaN;
  hplot2 = Rplot*par.Hcr;
 
  %% Update the tracking fields and save if...
  htrack(:,n) = h(:,2);
  Rtrack(:,n) = R(:,2);
  vtrack(:,n) = v_n(:)
  if (n==1 || n==par.nt || found_negative_h==1 || par.icelatlim==1 ...
      || par.icelatpole==1 || par.icelateq==1 ... || par.icelatstable==1)
      save(sprintf('%s/tracking-exp-%.2d-Q-%.2d-1d-sphere-nonlinear-resnum-%.2d-eps-%.2d-%s.mat',trackingfolder_FIS,par.EBM_expnum,par.Qo,par.N,100*var1,var2),'htrack(:,1:n)','Rtrack(:,1:n)','vtrack(:,1:n)');

  %% -----------
  %% plot h,u,v:
  %% -----------
  if (par.N==1 | floor(par.N/par.Nplot)*par.Nplot==par.N || par.N==par.Nt) & (n==1 || n==par.nt || found_negative_h==1 || par.icelatlim==1 ... 
       || par.icelatpole==1 || par.icelateq==1 ...
       || par.icelatstable==1)

    jb=2; je=par.nj-1;
    
    fig=figure(1); clf
    set(fig,'Visible','off');
    subplot(4,1,1) 
    %% note minus multiplying v, to chave from co-latitude to latitude:
    [ax,hl1,hl2]=plotyy(90-par.theta(je:-1:jb),hplot(je:-1:jb).*nan_mask(je:-1:jb) ...
             ,90-par.theta(je:-1:jb),-v_np1(je:-1:jb).*nan_mask(je:-1:jb)*par.year);
    set([hl1,hl2],'linewidth',2);
    vmax=par.year*max(abs(v_np1));
    %h1=xlabel(ax(1),'latitude');
    h2b=ylabel(ax(1),'h (m)');
    h2a=ylabel(ax(2),'v (m/yr)');
    h3=title(sprintf('hcr = %.2d,;v max=%3.0fm/yr',par.Hcr,vmax));
    set(ax(2),'xticklabel','');
    xlim(ax(1),[90-par.theta(je),90-par.theta(jb)]);
    xlim(ax(2),[90-par.theta(je),90-par.theta(jb)]);
    set([gca,h2a,h2b,h3],'fontsize',10);
    set_axis_limits(ax(1),90-par.theta(je:-1:jb),hplot(je:-1:jb));
    set_axis_limits(ax(2),90-par.theta(je:-1:jb),v_np1(je:-1:jb)*par.year);
    %% remove double tick marks on right y-axis:
    set(gca,'box','off'); %remove the box
    set(ax(2),'XAxisLocation','top','linewidth',1,'XTickLabel',[]) %cover the top box manually
    %%set(ax(2),'YTick',[]) %remove the ticks
    %% end of removing double tick marks
    
    subplot(4,1,2)
    hl1=plot(90-par.theta(je:-1:jb),T_surface(je:-1:jb,2)-par.T_f,90-par.theta(je:-1:jb),T_ocean(je:-1:jb,2)-par.T_f,':');
    set(hl1,'linewidth',1.5);
    grid on
    xlim([90-par.theta(je),90-par.theta(jb)]);   %% xlim([xmin xmax])
    %%h1=xlabel('latitude');
    legend Ts To
    h2=ylabel('(C)');
    h3=title('Surface Temperature');
    set([gca,h2,h3],'fontsize',10);


    subplot(4,1,3)
    PLOT=zeros(par.nj);
    hl1=plot(90-par.theta(je:-1:jb) ...
             ,S(je:-1:jb).*par.s(je:-1:jb)'.*nan_mask(je:-1:jb)*par.year,'r');
    hold on
    hl2=plot(90-par.theta(je:-1:jb) ...
             ,div_hv_n(je:-1:jb).*par.s(je:-1:jb)'.*nan_mask(je:-1:jb)*par.year,'--g');
    hl3=plot(90-par.theta(je:-1:jb) ...
             ,kappa_del2_h_n(je:-1:jb).*par.s(je:-1:jb)'.*nan_mask(je:-1:jb)*par.year,'--c');
    rhs=div_hv_n-kappa_del2_h_n;
    hl4=plot(90-par.theta(je:-1:jb) ...
             ,rhs(je:-1:jb).*par.s(je:-1:jb)'.*nan_mask(je:-1:jb)*par.year,'--b');
    xlim([90-par.theta(je),90-par.theta(jb)]);   %% xlim([xmin xmax])
    h1=xlabel('latitude');
    h2=ylabel('m/yr');
    h3=title(sprintf('terms in h eqn'));
    h4=legend('S','\nabla vh','k\nabla^2h','rhs','Location','southeast');
    set([hl1,hl2,hl3,hl4],'linewidth',2)
    set([gca,h1,h2,h3],'fontsize',10);
    set([h4],'fontsize',8);
    %pause
    
    subplot(4,1,4)
    %% note minus multiplying v, to chave from co-latitude to latitude:
    [ax,hl1,hl2]=plotyy(90-par.theta(je:-1:jb),Rplot(je:-1:jb),90-par.theta(je:-1:jb),hplot2(je:-1:jb));
    %set([hl1],'Line','--');
    hl1.LineStyle = ':';
    hl2.LineStyle = '--';
    set([hl1,hl2],'linewidth',2);
    h1=xlabel(ax(1),'latitude');
    h2b=ylabel(ax(1),'R');
    h2a=ylabel(ax(2),'h (m)');
    h3=title('Fractional Ice Cover (R) and height if uniformly spread over cell');
    if par.Nedge~=0
      lab1 = par.theta(par.Nedge);
    else 
      lab1 = NaN;
    end
    if par.Sedge~=0
      lab2 = par.theta(par.Sedge);
    else
      lab2 = NaN;
    end
 
   %% plot effective viscosity:
    fig_B=figure(3); clf; 
    set(fig_B,'Visible','off');
    Bplot=B*par.R./hplot; Bplot(mask==0)=NaN; % hplot was h(:,3)
    plot(90-par.theta(jb:je),log10(Bplot(je:-1:jb))');
    h1=xlabel('latitude');
    h2=ylabel('log_{10}(B)');
    h3=title(sprintf('log_{10}(B), n=%d, t=%gkyr',n,time_kyr));
    set([gca,h1,h2,h3],'fontsize',10);


    % save final figures
    set(fig, 'PaperUnits', 'inches'); set(fig, 'PaperSize', [5.5 6]);
    set(fig, 'PaperPosition', [0 0 5.5 6]); % [left, bottom, width, height];

    saveas(fig,sprintf('%s/main-output-%.2d.pdf',plotfolder,n));
    %saveas(fig,sprintf('%s/main-output-%.2d.fig',plotfolder,n));

    set(fig_B, 'PaperUnits', 'inches'); set(fig, 'PaperSize', [4 4]);
    set(fig_B, 'PaperPosition', [0 0 4 4]); % [left, bottom, width, height];

    saveas(fig_B,sprintf('%s/B-%.2d.pdf',plotfolder,n)); 
    %saveas(fig_B,sprintf('%s/B-%.2d.fig',plotfolder,n));


    % Figure 10: Source Term
    fig = figure(10); clf
    set(fig,'Visible','off');
    plot(90-par.theta(je:-1:jb),S(je:-1:jb)*par.year*100,'-+k',90-par.theta(je:-1:jb),dhdt_cond(je:-1:jb)*par.year*100,'--r',90-par.theta(je:-1:jb),dhdt_odiff(je:-1:jb)*par.year*100,'--b',90-par.theta(je:-1:jb),S_init(je:-1:jb)*par.year*100,'--g');
    yl = ylabel('Source Term (cm/yr)');
    xl = xlabel('Latitude');
    tl = title('Source Terms');
    legend Stot C Odiff S-EBM
    set([gca,xl,yl,tl],'fontsize',10); 

    saveas(gcf,sprintf('%s/source-%.2d.png',plotfolder,n));

    %pause
    if found_negative_h==1
      return
    end
  end %% plotting
  
  %% Write restart
  % Caution: this code rewrites over existing fields for export
  %floor(n/par.nwrite_restart)*par.nwrite_restart==n ||
  if n==par.nt || par.icelatlim==1 || par.icelatpole==1 || par.icelateq==1|| par.icelatstable==1

    theta=par.theta;
    phi=par.phi;

    % Update h(:,3),R(:,3) before writing Restart
    new_sg=find((h(:,3)<par.Hcr) & (h(:,3)>0));
    if length(new_sg)~=0
      R(new_sg,3) = find_R(h(new_sg,3),par);
      for i = 1:length(new_sg)
	j = new_sg(i);
	h(j,3) = 0;
      end
    end
   
    % Output FISEBM time-series fields
    par.h_min = min(h(find(h(:,3)>0),3));
    par.h_max = max(h(:,3));

    index = find(R(:,3)>0);
    indexlen = length(index);
    hx = zeros(indexlen,1);
    if indexlen>0
      for i = 1:indexlen
	j = index(i);
	if R(j,3)==1
	  hx(j) = h(j,3);
	elseif R(j,3)>0
	  hx(j) = R(j,3)*par.Hcr;
	end
      end 
    end
    % average ice thickness (not area weighted)
    if indexlen>0
      par.h_ave = sum(hx)/length(hx);
    else
      par.h_ave = 0;
    end

    index = find(par.R_initial>0);
    indexlen = length(index);
    hx = zeros(indexlen,1);
    if indexlen>0
      for i = 1:indexlen
	j = index(i);
	if R(j,3)==1
	  hx(j) = par.h_initial(j);
	elseif R(j,3)>0
	  hx(j) = par.R_initial(j)*par.Hcr;
	end
      end 
    end
    % average ice thickness (not area weighted)
    if indexlen>0
      par.h_init_ave = sum(hx)/length(hx);
    else
      par.h_init_ave = 0;
    end

    % latitudinal ice extent
    par.h_end = h(:,3);
    par.R_end = R(:,3);

    % mean ocean temp
    par.To_ave = globmean(To(:,3),par);
    % max ocean temp
    par.To_max = max(To(:,3));
    % min ocean temp
    par.To_min = min(To(:,3));  

    % mean atmos temp
    par.Ta_ave = globmean(Ta(:,3),par);
    % max atmos temp
    par.Ta_max = max(Ta(:,3));
    % min atmos temp
    par.Ta_min = min(Ta(:,3));

    % mean velocity
    vindex = find(isnan(v_np1)==0);
    vsum = 0;
    vsum = nansum(abs(v_np1(vindex)));
    if length(vindex)>0
      par.v_ave = vsum/length(vindex)*par.year;
    else 
      par.v_ave = 0;
    end

    par.v_max = max(abs(v_np1))*par.year;
    par.v_min = min(v_np1(find(v_np1>0)))*par.year;

    Ts(:,1:2) = T_surface(:,1:2);

    To(:,1:2) = T_ocean(:,1:2);

    restartfolder_FIS='/u/scratch/f/fspauldi/exosnowballs/FISRestart';

    save(sprintf('%s/restart-exp-%.2d-Q-%.2d-1d-sphere-nonlinear-resnum-%.2d-eps-%.2d-%s.mat',restartfolder_FIS,par.EBM_expnum,par.Qo,par.N,100*var1,var2),'h','R','Ta','To','Ts','qa','v_n','mask','theta','phi','B');
    
    if par.icelatlim==1 || par.icelatpole==1 || par.icelateq==1 || par.icelatstable==1
      return;
    end
 
  end

  %% prepare for next time step:
  h(:,1)=h(:,2);
  h(:,2)=h(:,3);
  R(:,1)=R(:,2);
  R(:,2)=R(:,3);

  T_surface(:,1) = T_surface(:,2);
  T_ocean(:,1) = T_ocean(:,2);

  %% Reset h and R 
  h(:,3) = 10000; 
  R(:,3) = NaN;  

  clear exsgi isogci oogci sgi;
  exsgilen = 0;
  isogcilen = 0;
  oogcilen = 0;
  sgilen = 0;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function set_axis_limits(h,x,y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set axis limits tighter around min/max of y data than the
%% typical Matlab default.
%% If plotting more than one array using more than one plot call, use:
%% set_axis_limits(gca,x,[y1,y2])
%% If y1,y2 are column vectors, use
%% set_axis_limits(gca,x,[y1',y2'])
%% If wish to add min/max values in addition to the data (e.g. if
%% data are a constant, and a wider range is desired):
%% set_axis_limits(gca,x,[ymin,ymax,y1,y2])

%% set y limits:
y=reshape(y,1,[]); % returns a 1xN vector of y
V=axis; % returns x-axis and y-axis limits for current axes
a1=min(y); 
a2=max(y);
V(3)=a1-(a2-a1)*0.15; 
V(4)=a2+(a2-a1)*0.15;
if V(3)==V(4); 
  V(3)=a1*0.9;
  V(4)=a2*1.1;
  if V(3)==0.0
    V(3)=-1;
    V(4)=1;
  end
end
%% set x limits:
V(1)=x(1);
V(2)=x(length(x));

axis(h,V);
set(h,'YTickMode','auto');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function par = EBM(EBM_expnum,N,Qo,var1,var2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Solve the energy balance model for ocean, atmosphere, and ice 
%% Francisco, 2018

par=set_EBM_parameters(EBM_expnum,N,Qo,var1,var2);

if strcmp(par.model_to_run,'2d-sphere')
  %% our 2D energy balance model using spherical coordinates:
  par = integrate_EBM_1d_sphere(par,var2);
else
  disp('*** no such model to run.');
  return
end

fprintf(1,'done.\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function par=set_EBM_parameters(EBM_expnum,N,Qo,var1,var2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

par.alpha_o = 0.25; % ocean surface albedo (see A13.1)
par.alpha_i = 0.6; % check with Eli about ice albedo 

% par.Hcr defined in EBMInput file

par.N = N;
par.EBM_expnum = EBM_expnum;
par.Nt = 500;
if par.N>5
  par.Nplot = 5;
else
  par.Nplot = 1;
end

par.icelatlim = 0;
par.icelatpole = 0;
par.ebalance = 0;


% XX - atmospheric layer absorptivity (A_a)
if EBM_expnum==1
  %par.A_a = 0.6;
  par.A_a = var1;
  par.version = '1d-sphere-global';
elseif EBM_expnum==2
  par.A_a = 0.63;
  par.version = '1d-sphere-partial';
elseif EBM_expnum==3
  par.A_a = var1;
  %par.A_a = 0.94;
  par.version = '1d-sphere-icefree';
elseif EBM_expnum==4
  par.A_a = 0.94;
  par.version = '1d-sphere-modern';
end 

par.Beta = 7; %ocean-ice base heat flux coefficient, W/(m^2 K)
%par.Beta = 50; %ocean-ice base heat flux coefficient, W/(m^2 K)
par.c_a = 1004.64; % specific heat of air, J/(kg K) 
par.c_o = 4218; % specific heat of water, J/(kg K)
par.c_i = 2106; % specific heat of ice J/(kg K)

% C_D :  neutral drag coefficient (A13.3)
par.k = 0.4; % von Karman constant
par.v_a = 5; % ~40m level wind speed
par.z_a = 40; % reference height
par.z_s = 0.001; % surface roughness length for ice
par.z_o = 0.0001; % surface roughness length for ocean

%
par.CD_o = (par.k/(log(par.z_a/par.z_o)))^2;  
%par.CD_s = (par.k/(log(par.z_a/par.z_s)))^2;
par.CD_s = par.CD_o;

par.CD_evap = par.CD_o*1.65;
par.CD_sens = par.CD_o/4.5;

par.d_i = 0.5; % thickness of surf seasonal thermal layer (A13.4)
% Diffusivities (m^2 s^-1)
%par.D_a = 1.3e6; % atmospheric heat diffusitity 
%par.D_q = 1.69e6; % atmospheric water vapor diffusivity  (A13.5) 
%par.D_a = 0.7e6; % XX
par.D_a = 1.3e6;
par.D_q = 1.69e6;

%par.D_o = 2.6e4; % ocean heat diffusivity
par.D_o = 3e4; % XX 8e5

par.G = 0.06; % geothermal heat flux, W/m^2

par.h_a = 8400; % atmospheric thickness for heat, m
par.h_q = 6000; % atmospheric thickness for water vapor
par.h_o = 50; % ocean mixed layer thickness

par.k_s = 0.2; % snow thermal conductivity, W/(mK)
par.k_i = 2.1; % ice thermal conductivity, W/(mK)
% Latent Heat (J/kg)
par.L_f = 0.334e6; % latent heat of H20 fusion 
par.L_v = 2.5104e6; % latent heat of H20 vaporization
par.L_s = 2.8440e6; % latent heat of H20 sublimation
% Densities (kg/m^3)
par.rho_a = 10^5/(287.04*287);
par.rho_o = 1000; % density of liquid water
par.rho_i = 900; % density of ice
par.rho_s = 250; % density of snow
par.sigma = 5.66961e-8; % Stefan-Boltzmann constant, W/(m^2 K^-4)
par.tau_p = 12*86400; % timescale for precipitation rate (s)
par.nn = 3; % Exponent of Glen's Law

par.R_d = 287.058; % specific gas constant for dry air, J/(kg K) 
par.R_v = 461.5; % specific gas constant for water vapor
par.epsilon = par.R_d/par.R_v;

par.Ta_o = 260; % 273 
par.To_o = 290; % Average ocean temperature
par.Ts_o = 260; % Average surface temperature
par.p_a = 10^5; % Atmospheric pressure level
par.qa_o = 0.01; % Specific humidity

%par.Hcr, should be specified in Input file 

par.g=10; % gravity
par.T_f = 273.15; % melt freeze point of ice/water (K)

%% (J/kg, from wikipedia's latent_heat), 
%% http://en.wikipedia.org/wiki/Latent_heat#Table_of_latent_heats :
par.mu=par.rho_i/par.rho_o;
par.year=365*86400;
par.day=86400;

par.R=6300e3; % Earth Radius

% Thickness diffusivity, numerical only, make it as small as possible:
par.kappa=1.2e0;    % 1.0e0
par.kappa_2d=1.2e0;  % 1.0e0
par.T_surface_profile_type='warm';

% Units of Time
if par.N>1
  par.tfac=5;
else
  par.tfac=10;
end
par.Time=par.tfac*par.year; % time to run the model for 
%par.dt=60*60*2; 
par.dt = 30*60;
%par.dt = 60*60;
par.nt=ceil(par.Time/par.dt); % par.nt = 2000

par.delta_T = 0.0001;

par.nplot=12*365*5; %48*7*54; % plot every x time steps
par.nplot_2d=100;

par.ni=89;
par.nj=89;
par.nk=42;
par.EQ = ceil(par.nj/2);

par.EBM_expnum=EBM_expnum;

%% Section: read scenario parameters (Input File)
addpath EBMInput;  % Could also put this line in my ~/matlab/startup.m
eval(sprintf('exp_%.2d',EBM_expnum));

%% physical domain, including boundary points, is [2:nj-1,2:nk-1]
par.L = 2e7;
par.dzeta=1/(par.nk-3);
par.dx=par.L/(par.ni-3);
par.dy=par.L/(par.nj-3);

par.zeta =-par.mu+([1:par.nk]-2)*par.dzeta;
par.x=([1:par.ni]-2)*par.dx;
par.y=([1:par.nj]-2)*par.dy;

%% spherical coordinates:
par.theta_north=10;
par.dphi=360/(par.ni-3);
par.dtheta=(90-par.theta_north)*2/(par.nj-3);
par.dphi_rad=par.dphi*pi/180;
par.dtheta_rad=par.dtheta*pi/180;
par.phi=([1:par.ni]-2)*par.dphi;
par.theta=par.theta_north+([1:par.nj]-2)*par.dtheta;
par.s=sind(par.theta);
par.c=cosd(par.theta);


% Abbot et al., 2011
par.s2 = -0.482;
par.S = 1 - par.s2/2 + 1.5*par.s2*par.c.^2;
par.Qo = Qo; 
par.Q = par.Qo/globmean(par.S',par); % solar constant on Earth

% Build new EBM restart files
if par.N==1
  mkrestart(par,var2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h,R,Ta,To,Ts,qa]=EBM_read_restart(par,var2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% which_restart can be '1d-sphere-global','1d-sphere-partial','1d-sphere-icefree', '1d-sphere-modern'
h = zeros(par.nj,3);
To = h; Ta = h; qa = h; R=h;
Ts = NaN(par.nj,3);

which_restart = par.version;

restartfolder_FIS = '/u/scratch/f/fspauldi/exosnowballs/FISRestart';
restartfolder_EBM = '/u/scratch/f/fspauldi/exosnowballs/EBMRestart';

if par.N==1 && par.EBM_expnum<=4
  fprintf(1,'Initializing EBM @ N = 1 with mkrestart file.\n');
  restart_filename = sprintf('%s/mkrestart-EBM-%s-Q-%.2d-eps-%.2d-%s.mat',restartfolder_EBM,which_restart,par.Qo,100*par.A_a,var2);
else 
  fprintf(1,'Initializing EBM @ N = %.2d with FIS restart file.\n',par.N);
  restart_filename = sprintf('%s/restart-exp-%.2d-Q-%.2d-1d-sphere-nonlinear-resnum-%.2d-eps-%.2d-%s.mat',restartfolder_FIS,par.EBM_expnum,par.Qo,par.N-1,100*par.A_a,var2)
end

if exist(restart_filename,'file')
  load(restart_filename);
  Ts(:,3) = NaN;
  To(:,3) = 0;
  %qa(:,:) = 0;
else
  fprintf(2,'EBM error: no mkrestart or FIS restart file %s.\n',restart_filename);
  keyboard;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mkrestart(par,var2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build your own restart file to test the EBM
%addpath('/home/francisco17/ExoSnowballs/ver02/FISEBM/EBMRestart/');
addpath('/u/home/f/fspauldi/exosnowballs/ver02/Insolation');

restartfolder_FIS = '/u/scratch/f/fspauldi/exosnowballs/FISRestart';
restartfolder_EBM = '/u/scratch/f/fspauldi/exosnowballs/EBMRestart';

if par.EBM_expnum==1
  which_restart = '1d-sphere-global';
elseif par.EBM_expnum==2
  which_restart = '1d-sphere-partial';
elseif par.EBM_expnum==3
  which_restart = '1d-sphere-icefree';
elseif par.EBM_expnum==4
  which_restart = '1d-sphere-modern';
else
  fprintf(2,'error: invalid EBM_expnum\n');
  return
end

h = zeros(par.nj,3);
To = h; Ta = h; qa = h; R = h;
Ts = NaN(par.nj,3);

if strcmp('1d-sphere-global',which_restart)==1
  h = h+50;
  To(:,:) = par.T_f;
  Ta(:,3) = Ta_surface_function(par);
  Ta(:,2) = Ta(:,3);
  Ta(:,1) = Ta(:,3);
  Ts(:,1:2) = Ta(:,1:2) - 10;
  qa(:,:) = 0;
  R = ones(par.nj,3);
  save(sprintf('%s/mkrestart-EBM-%s-Q-%.2d-eps-%.2d-%s.mat',restartfolder_EBM,which_restart,par.Qo,100*par.A_a,var2),'h','qa','R','Ta','To','Ts');
elseif strcmp('1d-sphere-partial',which_restart)==1
  addpath EBMInput;  
  eval(sprintf('exp_%.2d',par.EBM_expnum)); % caution about this line

  h([1:par.Nedge,par.Sedge:end],:) = 1000;
  To(:,:) = par.T_f;
  Ta(:,3) = Ta_surface_function(par);
  Ta(:,2) = Ta(:,3);
  Ta(:,1) = Ta(:,3);
  Ts([1:(par.Nedge+10),(par.Sedge-10):end],1:2) = Ta([1:(par.Nedge+10),(par.Sedge-10):end],1:2) - 10;
  qa(:,:) = 0;
  R([1:par.Nedge,par.Sedge:end],:) = 1;
  R((par.Nedge+1):(par.Nedge+10),:) = 0.5;
  R((par.Sedge-10):(par.Sedge-1),:) = 0.5;

  save(sprintf('%s/mkrestart-EBM-%s-Q-%.2d-eps-%.2d-%s.mat',restartfolder_EBM,which_restart,par.Qo,100*par.A_a,var2),'h','qa','R','Ta','To','Ts');
elseif strcmp('1d-sphere-icefree',which_restart)==1
  addpath EBMInput;  
  eval(sprintf('exp_%.2d',par.EBM_expnum)); % caution about this line
  h(1:par.Nedge,:) = 10;
  h(par.Sedge:end,:) = 10;
  To(:,:) = par.T_f;
  Ta(:,3) = Ta_surface_function(par)+70;
  Ta(:,2) = Ta(:,3);
  Ta(:,1) = Ta(:,3);
  Ts(:,:) = NaN;
  qa(:,:) = 0;
  Ts([1:par.Nedge,par.Sedge:end],1:2) = Ta([1:(par.Nedge),(par.Sedge):end],1:2) - 10;
  R([1:par.Nedge,par.Sedge:end],:) = 1;

  save(sprintf('%s/mkrestart-EBM-%s-Q-%.2d-eps-%.2d-%s.mat',restartfolder_EBM,which_restart,par.Qo,100*par.A_a,var2),'h','qa','R','Ta','To','Ts');
elseif strcmp('1d-sphere-modern',which_restart)==1 
  addpath EBMInput;  
  eval(sprintf('exp_%.2d',par.EBM_expnum)); % caution about this line
  h(1:par.Nedge,:) = 2000;
  h(par.Sedge:end,:) = 2000;
  To(:,:) = par.T_f;
  Ta(:,3) = Ta_surface_function(par)+70;
  Ta(:,2) = Ta(:,3);
  Ta(:,1) = Ta(:,3);
  Ts([1:par.Nedge,par.Sedge:end],1:2) = Ta([1:(par.Nedge),(par.Sedge):end],1:2) - 10;
  qa(:,:) = 0;
  R([1:par.Nedge,par.Sedge:end],:) = 1;
  save(sprintf('%s/restart-EBM-%s-Q-%.2d-eps-%.2d-%s.mat',restartfolder_EBM,which_restart,par.Qo,100*par.A_a,var2),'h','qa','R','Ta','To','Ts');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Qa=Q_a(R,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Qa: net solar radiative flux absorbed by the atmosphere
Qa = zeros(par.nj,1);
% atmosphere is transparent to SW radiation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Qo=Q_o(R,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Qo: net solar radiative flux absorbed by ocean
Qo = zeros(par.nj,1);
for j = 1:par.nj
  if R(j)<1
    Qo(j) = (1-par.alpha_o)*par.S(j)*par.Q/4;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Qs=Q_s(R,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Qs: net solar radiative flux absorbed by ice surface
[r,c] = size(R);
Qs = zeros(r,c);

for j = 1:par.nj
  if R(j)>0
    Qs(j) = (1-par.alpha_i)*par.S(j)*par.Q/4;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Ia=I_a(R,par,Ta,To,Ts,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ia: net thermal infrared fluxes absorbed by atmosphere
Ia = zeros(par.nj,1);

Ts(isnan(Ts)) = 0;

for j = 1:par.nj
  % area weighted mean
  Ia(j) = (1-R(j))*par.A_a*par.sigma*To(j)^4 ...
          + R(j)*par.A_a*par.sigma*Ts(j)^4 ...
          - 2*par.A_a*par.sigma*Ta(j)^4;
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Io=I_o(R,Ta,To,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Io: net thermal infrared fluxes absorbed by ocean
Io = zeros(par.nj,1);
% ocean emissivity = 1
for j = 1:par.nj
  if R(j)<1
    % ocean
    Io(j) = par.A_a*par.sigma*Ta(j)^4 - par.sigma*To(j)^4;
  end
end
% subgrid weightings must be applied outside this helper function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Is=I_s(R,Ta,Ts,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Is: net thermal infrared fluxes absorbed by ice surface
Is = zeros(par.nj,1);

Ts(isnan(Ts))=0;

for j = 1:par.nj
  if R(j)>0
    Is(j) = par.A_a*par.sigma*Ta(j)^4 - par.sigma*Ts(j)^4;
  end
end
% subgrid weightings must be applied outside this helper function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Ha=H_a(R,Ho,Hs,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sensible heat fluxes from overall surface to atmosphere
Ha = zeros(par.nj,1);
for j = 1:par.nj
  % area-weighted mean
  Ha(j) = -((1-R(j))*Ho(j) + R(j)*Hs(j));
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Ho=H_o(R,Ta,To,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sensible heat fluxes from the atmosphere to the ocean
Ho = zeros(par.nj,1);
rhoa = zeros(par.nj,1);

rhoa = rho_a(Ta,par);
for j = 1:par.nj
  if R(j)<1
    Ho(j) = -rhoa(j)*par.c_a*par.CD_sens*(To(j)-Ta(j));
  end
end
% subgrid weightings must be applied outside this helper function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Hs=H_s(R,Ta,Ts,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sensible heat fluxes from the atmosphere to the ice surface
Hs = zeros(par.nj,1);
rhoa = zeros(par.nj,1);

rhoa = rho_a(Ta,par);

for j = 1:par.nj
  if R(j)>0
    Hs(j) = -rhoa(j)*par.c_a*par.CD_sens*(Ts(j)-Ta(j));
  end
end
% subgrid weightings must be applied outside this helper function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function En=E(R,Ta,To,Ts,qa,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evaporation rate from overall surface to the atmosphere
En = zeros(par.nj,1);
dq_To = zeros(par.nj,1);
dq_Ts = zeros(par.nj,1);
rhoa = zeros(par.nj,1);

Ts(find(isnan(Ts)==1)) = 0;

dq_To = qa - q_sat(To,par);
dq_Ts = qa - q_sat(Ts,par);
rhoa = rho_a(Ta,par);

for j = 1:par.nj
  En(j) = -par.CD_evap*rhoa(j)*((1-R(j))*dq_To(j) + R(j)*dq_Ts(j));
  if En(j)<0
    En(j) = 0;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Pn=P(R,Ta,Ts,qa,par,plotfolder,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Precipitation rate P 
% rho_a is temperature dependent

Pn = zeros(par.nj,1);
ra = zeros(par.nj,1); % atmospheric relative humidity
e = zeros(par.nj,1); % water vapor partial pressure
ew = zeros(par.nj,1); % saturation vapor pressure
rhoa = zeros(par.nj,1);
ra_max = 0.8;

rhoa = rho_a(Ta,par);
e = e_p(qa,par);
ew = e_w(Ta,par); %xx
ew = mb_2_Pa(ew);
ra = r_a(e,ew);

index = find(ra>ra_max);
indexlen = length(index);

for i = 1:indexlen
  j = index(i);
  Pn(j) = (rhoa(j)*par.h_q*qa(j)/par.tau_p)*(ra(j)^3 - ra_max^3);

  if Pn(j)<0
    Pn(j) = 0;
    fprintf(1,'error: Pn < 0\n');
    keyboard;
  end
end

if (par.N==1 || floor(par.N/par.Nplot)*par.Nplot==par.N || par.N==par.Nt) & n==par.nt
%if n==par.nt 
   je = par.nj-1;
   jb = 2;

   Pi = P_i(Pn,Ts,par);
   prec_i = Pi*par.year*100/par.rho_i;
   prec_t = Pn*par.year*100/par.rho_i;

   % Figure 1: Precipitation
   fig = figure(1); clf
   set(fig,'Visible','off');
   subplot(4,1,1)
   plot(90-par.theta(je:-1:jb),prec_i(je:-1:jb),'-or',90-par.theta(je:-1:jb),prec_t(je:-1:jb),'-b');
   yl = ylabel('Precipitation  (cm/yr)');
   xl = xlabel('Latitude');
   tl = title('Precipitation @ t=n');
   legend Pi Ptotal
   set([gca,xl,yl,tl],'fontsize',10); 

   subplot(4,1,2)
   plot(90-par.theta(je:-1:jb),rhoa(je:-1:jb));
   yl = ylabel('Atmospheric Density (kg/m^3)');
   xl = xlabel('Latitude');
   tl = title('Atmospheric Density');
   set([gca,xl,yl,tl],'fontsize',10);

   subplot(4,1,3)
   plot(90-par.theta(je:-1:jb),qa(je:-1:jb));
   yl = ylabel('Atmospheric Specific Humidity');
   xl = xlabel('Latitude');
   tl = title('Specific Humidity');
   set([gca,xl,yl,tl],'fontsize',10);

   subplot(4,1,4)
   plot(90-par.theta(je:-1:jb),ra(je:-1:jb));
   yl = ylabel('Atmospheric Relative Humidity');
   xl = xlabel('Latitude');
   tl = title('Relative Humidity');
   set([gca,xl,yl,tl],'fontsize',10);

   saveas(gcf,sprintf('%s/Pn-comparison-%.2d.png',plotfolder,n)); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mb2Pa=mb_2_Pa(p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% convert mb=hPa to Pa
mb2Pa = p*100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Pa2mb=Pa_2_mb(p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% convert Pa to hPa=mb
Pa2mb = p/100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ra=r_a(e,ew)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% atmospheric relative humidity
% all inputs must be in Pa
% ew_o output in mb, so convert to Pa first
% ew_s output in Pa
[r,c] = size(e);
ra = zeros(r,c);

ra = (e./ew); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rho=rho_a(Ta,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Temperature-dependent atmospheric density
[r,c] = size(Ta);
rho = zeros(r,c);

rho = par.p_a./(par.R_d*Ta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function e=e_p(qa,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% partial pressure of water vapor (Pa)
[r,c] = size(qa);
x = zeros(r,c);

x = par.epsilon^(-1)*qa./(1-qa);
e = par.p_a*x./(1+x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ew=e_w(Ta,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saturation water vapor pressure (mb) from Emmanuel 4.4.14
% p 116-117
% over a planar surface of liquid water
[r,c] = size(Ta);
TT = zeros(r,c);

TT = Ta - par.T_f;
ew = 6.112*exp((17.67*TT)./(TT+243.5));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function ew=ew_o(Ta,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saturation water vapor pressure (mb) from Emmanuel 4.4.14
% p 116-117
% over a planar surface of liquid water
%[r,c] = size(Ta);
%TT = zeros(r,c);

%TT = Ta - par.T_f;
%ew = 6.112*exp((17.67*TT)./(TT+243.5));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function ew=ew_s(Ta,par,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Saturation water vapor pressure (Pa) over a planar surface of ice
% from Murphy and Koop, 2005 (~500 citations)
% Range: Ti > 110 K
%ew = zeros(par.nj,1);

%domain = find(isnan(Ta)==0);

%for i = 1:length(domain)
  %j = domain(i);
  %ew(j) = exp(9.550426 - 5723.265./Ta(j) + 3.53068*log(Ta(j)) - 0.00728332*Ta(j));
  % ew in pascals
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function qsat=q_sat(Ta,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% saturation specific humidity over water
% Pa in mb, T in K
% first saturation mixing ratio over water (gr water vapor per gram dry air):
[r,c] = size(Ta);
rw = zeros(r,c);

rw = r_w(Ta,par);
% and from that, saturation specific humidity (gr water vapor per gram m oist air):
qsat = rw./(1+rw);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function qsat=qsat_o(Ta,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% saturation specific humidity over water
% Pa in mb, T in K
% first saturation mixing ratio over water (gr water vapor per gram dry air):
%[r,c] = size(Ta);
%rw = zeros(r,c);

%rw = rw_o(Ta,par);
% and from that, saturation specific humidity (gr water vapor per gram m oist air):
%qsat = rw./(1+rw);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function qsat=qsat_s(Ta,par,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% saturation specific humidity over ice
% Temp in K
% first saturation mixing ratio over ice (gr water vapor per gram dry air):
%rw = rw_s(Ta,par,n);

% and from that, saturation specific humidity over ice (gr water vapor per gram moist air):
%qsat = rw./(1+rw);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rw=r_w(Ta,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% saturation mixing ratio over water
% Pa in mb, bc so is ew(T_a)
% T in kelvin, as that's what required by ew(T_a)
% output mixing ratio is gr/gr.
[r,c] = size(Ta);
rw = zeros(r,c);
ew = e_w(Ta,par);

P = Pa_2_mb(par.p_a);
rw = par.epsilon*ew./(P-ew);
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function rw=rw_o(Ta,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% saturation mixing ratio over water
% Pa in mb, bc so is ew(T_a)
% T in kelvin, as that's what required by ew(T_a)
% output mixing ratio is gr/gr.
%[r,c] = size(Ta);
%rw = zeros(r,c);
%ew = ew_o(Ta,par);

%P = Pa_2_mb(par.p_a);
%rw = par.epsilon*ew./(P-ew);
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function rw=rw_s(Ta,par,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% saturation mixing ratio over ice
% Pressure in Pa
%  ew_s(Ts) input is Pa
% T in kelvin, as that's what required by ew_s(Ts)
% output mixing ratio is gr/gr.
%[r,c] = size(Ta);
%rw = zeros(r,c);
%ew = ew_o(Ta,par,n); % dorian edit

%P = par.p_a;
%rw = par.epsilon*ew./(P-ew);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Lv = L_v(Ta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% latent heat of vaporization of water
% input: T=temperature in Kelvin.
% Gill appendix 4, page 607
[r,c] = size(Ta);
TT = zeros(r,c);

TT = Ta-par.T_f;
Lv = 2.5008e6 - 2.3e3*TT;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function TS = Ta_surface_function(par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate Initial Surface Air Temperature Profile
y_middle = par.y(par.nj-1)/2;
lat = (par.y-y_middle)*0.5*pi/y_middle;
if strcmp(par.Ta_surface_profile_type,'mine')
  %% Initial Try
  TS=par.T_f+(-20-40*sin(pi*(abs(par.y)-0.5*par.y(par.nj-1))/par.y(par.nj-1)).^4);
elseif strcmp(par.Ta_surface_profile_type,'cold')
  %% Dorian's low CO2:
  TS=par.T_f-79+48*cos(lat).^2;
elseif strcmp(par.Ta_surface_profile_type,'warm')
  %% Dorian's high CO2:
  TS=par.T_f-58+47*cos(lat).^2;
else
  disp('*** no such surface temperature profile.');
  quit
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Eplot,netPrec,netEvap,Aplot,Surf,ToA] = ebalance(Ha_np1,par,R,Ta,To,Ts,dhdt_frz,dhdt_melt,n,C_np1,G_np1,P_np1,E_np1,rec,var2,Qo_np1,Qs_np1,qa)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper Function to check ToA and Surface Energy Balance
Ts(find(isnan(Ts)==1))=0;
% Net Shortwave (SW) at the surface
SW_net = (1-R).*Qo_np1 + (R).*Qs_np1;
% Net Longwave (LW) at the surface
LW_up = (1-R).*par.sigma.*To.^4 + R.*par.sigma.*Ts.^4;
LW_down = par.A_a*par.sigma*Ta.^4;
LW_net = LW_down - LW_up; 
% Sensible Heat (SH) from the surface to the atmosphere
SH = Ha_np1;
% Latent Heat (LHe) from the surface to the atmosphere
LHe = E_np1*par.L_v;
% Conductive Flux (C) into the ice surface
C = (R).*C_np1;
% Geothermal Flux (G) into the ocean mixed layer (regardless of ice cover)
G = G_np1.*(1-R);
% Surface Melting or Basal Freezing (F)
F_melt = par.rho_i*par.L_f*dhdt_melt; % dhdt_melt<0
F_frz = par.rho_i*par.L_f*dhdt_frz;   % dhdt_frz>0
F_net = F_melt + F_frz;
% Sensible Heat from the ocean to the ice 
% (zero in surface e bal for thick ice)
Rmask = ones(par.nj,1); index = find(R==1); Rmask(index)=0;
SHoi = R.*par.Beta.*(To-par.T_f).*Rmask;
%% Surface Energy Balance %% 
Surf = SW_net + LW_net - SH - LHe + C + G + F_net - SHoi; 

% Net Shortwave (SWtoa) at the top of atmosphere 
SWtoa = SW_net;
% Net Atmospheric Longwave (LWatm) to space
LWatm = par.A_a*par.sigma*Ta.^4;
% Net Surface Longwave (LWsfc) to space
LWsfc = (1-par.A_a)*LW_up;
% Latent Heat (LHp) of Condensation
%LHp = P_np1*par.L_v;
%LHp = zeros(par.nj,1);
% Residual P-E in the atmosphere
%PmE = par.L_v*(P_np1 - E_np1);

%% Top of Atmosphere Energy Balance %%
ToA = SWtoa - LWatm - LWsfc;

%% Outgoing Longwave Radiation (OLR) %%
% Downwelling Solar Insolation (SW_down)
SW_down = par.S'*par.Q/4;
% Upwelling Solar Insolation (SW_up)
SW_up = ((1-R)*par.alpha_o + (R)*par.alpha_i).*SW_down;
% Outgoing Longwave Radiation (OLR)
OLR = LWatm + LWsfc;

%% Global Mean Averages %%
netSurf = globmean(Surf,par);
netToA = globmean(ToA,par);
netPrec = globmean(P_np1,par);
netEvap = globmean(E_np1,par);
netLW_up = globmean(LW_up,par);
netSW_down = globmean(SW_down,par);
netSW_up = globmean(SW_up,par);
netOLR = globmean(OLR,par);

alpha_sfc = (1-R)*par.alpha_o + (R)*par.alpha_i;
Aplot = globmean(alpha_sfc,par);

%% Plotting %%
Eplot = zeros(1,2);
Eplot(1,1) = netToA;
Eplot(1,2) = netSurf;

if (par.N==1 || floor(par.N/par.Nplot)*par.Nplot==par.N || par.N==par.Nt) & n==par.nt

  plotfolder_EBM='/u/scratch/f/fspauldi/exosnowballs/EBMFigures';
  plotfolder = sprintf('%s/EBM-nonlinear-exp-%.2d-Q-%.2d-N-%.2d-eps-%.2d-%s',plotfolder_EBM,par.EBM_expnum,par.Qo,par.N,100*par.A_a,var2);
  
  jb = 2; je = par.nj-1;
  clf;
  fig = figure(1);
  set(fig,'Visible','off');
  subplot(2,1,1)
  plot(90-par.theta(je:-1:jb),ToA(je:-1:jb),'-+r',90-par.theta(je:-1:jb),SW_net(je:-1:jb),'k',90-par.theta(je:-1:jb),LWatm(je:-1:jb),'b',90-par.theta(je:-1:jb),LWsfc(je:-1:jb),'g');
  xl = xlabel('latitude');
  yl = ylabel('\DeltaW/m^2');
  tl = title(sprintf('ToA Energy Balance = %2.2d',Eplot(1)));
  legend ToA SW-net LWatm LWsfc
  set([gca,xl,yl,tl],'fontsize',10);

  subplot(2,1,2)
  plot(90-par.theta(je:-1:jb),Surf(je:-1:jb),'-+r',90-par.theta(je:-1:jb),SW_net(je:-1:jb),'k',90-par.theta(je:-1:jb),LHe(je:-1:jb),'b',90-par.theta(je:-1:jb),SH(je:-1:jb),'c',90-par.theta(je:-1:jb),LW_down(je:-1:jb),'y',90-par.theta(je:-1:jb),LW_up(je:-1:jb),'m',90-par.theta(je:-1:jb),G(je:-1:jb),90-par.theta(je:-1:jb),C(je:-1:jb),'g',90-par.theta(je:-1:jb),F_net(je:-1:jb),'--',90-par.theta(je:-1:jb),SHoi(je:-1:jb),'--');
  xl = xlabel('latitude');
  yl = ylabel('\DeltaW/m^2');
  tl = title(sprintf('Surface Energy Balance = %2.2d',Eplot(2)));
  legend Surf SW-net LHe SH LW-down LW-up G C F_mf SHoi
  annotation('textbox','String','Surf/ToA = SWnet - all');
  set([gca,xl,yl,tl],'fontsize',10);

  saveas(gcf,sprintf('%s/eBalance_lat_%.2d.png',plotfolder,n));
  
  clf;
  fig = figure(2);
  set(fig,'Visible','off');
  plot(90-par.theta(je:-1:jb),Surf(je:-1:jb),'-+r',90-par.theta(je:-1:jb),LHe(je:-1:jb),'b',90-par.theta(je:-1:jb),SH(je:-1:jb),'c',90-par.theta(je:-1:jb),G(je:-1:jb),90-par.theta(je:-1:jb),C(je:-1:jb),'g',90-par.theta(je:-1:jb),F_net(je:-1:jb),'--');
  %plot(90-par.theta(je:-1:jb),LHe(je:-1:jb),'b',90-par.theta(je:-1:jb),SH(je:-1:jb),'c',90-par.theta(je:-1:jb),G(je:-1:jb),90-par.theta(je:-1:jb),C(je:-1:jb),'g',90-par.theta(je:-1:jb),F_net(je:-1:jb),'--');
  xl = xlabel('latitude');
  yl = ylabel('\DeltaW/m^2');
  tl = title(sprintf('Surface Energy Balance = %2.2d',Eplot(2)));
  legend Surf LHe SH G C F_mf 
  annotation('textbox','String','Surf/ToA = SWnet - all');
  set([gca,xl,yl,tl],'fontsize',10);
 
  saveas(gcf,sprintf('%s/eBalance_extra_%.2d.png',plotfolder,n));

  %keyboard;

  if n==par.nt
    fprintf(1,'Energy Balance: @n=par.nt, global mean albedo = %.2d, netToA=%.2d & netSurf=%.2d & netPrec=%.2d & netEvap=%.2d & netSW_down=%.2d & netSW_up=%.2d & netOLR=%.2d\n',Aplot,netToA,netSurf,netPrec*100*par.year/par.rho_o,netEvap*100*par.year/par.rho_o,netSW_down,netSW_up,netOLR);
    if abs(netSurf)<0.5 & abs(netToA)<0.5
      fprintf(1,'Achieved Energy Balance\n');
    else
      fprintf(1,'Surface Energy Imbalance\n');
      %keyboard;
    end
  end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function eBalance_old(Ha_np1,par,R,Ta,To,Ts,dhdt_frz,dhdt_melt,n,C_np1,G_np1,P_np1,E_np1,rec,var2,Qo_np1,Qs_np1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper function to check ToA and Surface Energy Balance
Ts(find(isnan(Ts)==1))=0; % converts any NaN fields to 0 for calcs

alpha_sfc = (1-R)*par.alpha_o + R*par.alpha_i;
SW_net = (1-alpha_sfc).*par.Q.*par.S'/4; 

LW_up = (1-R).*par.sigma.*To.^4 + R.*par.sigma.*Ts.^4;

LW_down = par.A_a*par.sigma*Ta.^4;

SH_up = Ha_np1;

LH_up = E_np1*par.L_v;
    
dhdt = dhdt_frz + dhdt_melt;
F_M = -par.rho_i*par.L_f*dhdt;

F_a = par.A_a*par.sigma*Ta.^4; 

F_sfc = R.*(1-par.A_a)*par.sigma.*Ts.^4 + (1-R).*(1-par.A_a).*par.sigma.*To.^4;

OLR = F_a+F_sfc;
ToA = SW_net - OLR;
netToA = globmean(ToA,par);

Surf = SW_net - LH_up - SH_up + LW_down - LW_up + C_np1 - F_M; 
netSurf = globmean(Surf,par);

netPrec = globmean(P_np1,par);
netEvap = globmean(E_np1,par);

SW_down = par.S'*par.Q/4;
SW_up = alpha_sfc.*par.S'*par.Q/4;
%LW_up = OLR; error double definition

netSW_down = globmean(SW_down,par);
netSW_up = globmean(SW_up,par);
netOLR = globmean(OLR,par);


%if par.N==500 && (n==1 || floor(n/par.nplot)*par.nplot==n || n==par.nt)
%if (n==1 || floor(n/par.nplot)*par.nplot==n || n==par.nt)

if (par.N==1 || floor(par.N/par.Nplot)*par.Nplot==par.N || par.N==par.Nt) & (n==1 || floor(n/par.nplot)*par.nplot==n || n==par.nt)
  plotfolder = sprintf('EBMFigures/EBM-nonlinear-exp-%.2d-Q-%.2d-N-%.2d-eps-%.2d-%s',par.EBM_expnum,par.Qo,par.N,100*par.A_a,var2);
  jb = 2; je = par.nj-1;
  clf;
  fig = figure(1);
  set(fig,'Visible','off');
  subplot(2,1,1)
  plot(90-par.theta(je:-1:jb),ToA(je:-1:jb),'-+r',90-par.theta(je:-1:jb),SW_net(je:-1:jb),'k',90-par.theta(je:-1:jb),F_a(je:-1:jb),'b',90-par.theta(je:-1:jb),F_sfc(je:-1:jb),'g');
  xl = xlabel('latitude');
  yl = ylabel('\DeltaW/m^2');
  tl = title('ToA Energy Balance');
  legend ToA SW-net F-a F-sfc
  set([gca,xl,yl,tl],'fontsize',10);

  subplot(2,1,2)
  plot(90-par.theta(je:-1:jb),Surf(je:-1:jb),'-+r',90-par.theta(je:-1:jb),SW_net(je:-1:jb),'k',90-par.theta(je:-1:jb),LH_up(je:-1:jb),'b',90-par.theta(je:-1:jb),SH_up(je:-1:jb),'c',90-par.theta(je:-1:jb),LW_down(je:-1:jb),'y',90-par.theta(je:-1:jb),LW_up(je:-1:jb),'m',90-par.theta(je:-1:jb),C_np1(je:-1:jb),'g',90-par.theta(je:-1:jb),F_M(je:-1:jb),'--o');
  xl = xlabel('latitude');
  yl = ylabel('\DeltaW/m^2');
  tl = title('Surface Energy Balance');
  legend Surf SW-net LH-up SH-up LW-down LW-up C F_M
  annotation('textbox','String','Surf/ToA = SWnet - all');
  set([gca,xl,yl,tl],'fontsize',10);
  
  saveas(gcf,sprintf('%s/eBalance_lat_%.2d.png',plotfolder,n));

  if n==par.nt
    alpha_sfc = (1-R)*par.alpha_o + (R)*par.alpha_i;
    fprintf(1,'Energy Balance: @n=par.nt, global mean albedo = %.2d, netToA=%.2d & netSurf=%.2d & netPrec=%.2d & netEvap=%.2d & netSW_down=%.2d & netSW_up=%.2d & netOLR=%.2d\n',globmean(alpha_sfc,par),netToA,netSurf,netPrec,netEvap,netSW_down,netSW_up,netOLR);
    if abs(netSurf)<0.5 & abs(netToA)<0.5
      fprintf(1,'Achieved Energy Balance\n');
    else
      fprintf(1,'Surface Energy Imbalance\n');
      keyboard;
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Pi=P_i(Pn,Ts,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rate of Snowfall in prognostic ice h eq in kg/m^2/s
Pi = zeros(par.nj,1);
index = find(isnan(Ts)==0);

Pi(index) = Pn(index);
% subgrid weightings must be applied outside this helper function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C=Cflux(h,R,To,Ts,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Conductive Flux underneath sea ice
C = zeros(par.nj,1);
Ts(isnan(Ts))=0;

%if length(find(To~=par.T_f & R==1))>0
  %fprintf(1,'error: Cflux, To~=Tf under thick ice');
  %keyboard;
%end

for j=1:par.nj
  if R(j)==0
    C(j) = 0;
  elseif R(j)==1
    C(j) = par.k_i*(par.T_f - Ts(j))./h(j);
  else
    h_ice = par.Hcr; 
    C(j) = par.k_i*(par.T_f - Ts(j))./h_ice;
  end
end
% subgrid weightings must be applied outside of this helper function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dhdt=Cforc(h,R,To,Ts,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Conductive Basal Forcing of Sea Ice Melt/Freezing (used in ice flow model)
% The output90-par.theta(je:-1:jb),S(je:-1:jb)*par.year*100,'--ob', of this function has units of m/s.
%k_i = 10^4; % heat conductivity of ice (m^2/s) 

C = Cflux(h,R,To,Ts,par);
dhdt = C/(par.rho_i*par.L_f);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Fo=F_o(C,G,par,R)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Net Freezing or Melting Rate at Ice Base in prognostic ice h eq
% C is conductive flux under ice, G is the geothermal heat flux onto ice
Fo = zeros(par.nj,1);
for j = 1:par.nj
  if R(j)>0 
    Fo(j) = (C(j)-G(j))/(par.rho_i*par.L_f);
  else
    Fo(j) = 0;
  end
end
% subgrid weightins must be applied outside of this helper function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h,R,Ts,To,dhdt_melt]=Ts_check_old(h,par,qa,R,Ta,To,Ts,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Checks Ts for Unphysical Values and Melts Ice where Ts>273
dhdt = zeros(par.nj,1);
dh = zeros(par.nj,1);
dhdt_melt = zeros(par.nj,1);

Tsi = find(Ts>par.T_f);

if length(Tsi)>0
  Ts2 = ones(par.nj,1)*par.T_f;

  C = Cflux(h,R,To,Ts,par);
  Qs = Q_s(R,par);
  Is = I_s(R,Ta,Ts,par);
  Hs = H_s(R,Ta,Ts,par);
  Es = E_s(R,Ta,Ts,qa,par,n);

  C2 = Cflux(h,R,To,Ts2,par);
  Qs2 = Q_s(R,par);
  Is2 = I_s(R,Ta,Ts2,par);
  Hs2 = H_s(R,Ta,Ts2,par);
  Es2 = E_s(R,Ta,Ts2,qa,par,n);

  E = Qs + Is + Hs - par.L_v*Es + C; 
  E2 = Qs2 + Is2 + Hs2 - par.L_v*Es2 + C2; 

  dE = E - E2; 

  for i = 1:length(Tsi)
    j = Tsi(i); 

    dhdt(j) = dE(j)/(par.rho_i*par.L_f);
    dh(j) = dhdt(j)*par.dt;

    if dh(j)>0
      fprintf(1,'error: dh>0');
      keyboard;
    else 
      if R(j)==1
        H = h(j) - abs(dh(j));
        if H>=par.Hcr 
          % melts on grid scale
          h(j) = H;
          R(j) = find_R(h(j),par);
          Ts(j) = par.T_f;
          dhdt_melt(j) = dhdt(j);
        elseif H<par.Hcr & H>0
          % melts into sg scale
          R(j) = find_R(H,par);
          h(j) = 0; % update h
          Ts(j) = par.T_f; 
          dhdt_melt(j) = dhdt(j);
        else
          dhdt_melt(j) = -h(j)/par.dt;
          % melts completely, warming the ocean potentially
          h(j) = 0;
          R(j) = 0;
          Ts(j) = NaN;
          dE_used = dhdt_melt(j)*par.rho_i*par.L_f;
          dE_rem = abs(dE(j))-abs(dE_used);
          if dE_rem<0
            fprintf(1,'error_1:dE_rem<0');
            keyboard;
          end
          To(j) = To(j) + par.dt*(par.rho_o*par.c_o*par.h_o)^(-1)*dE_rem;
        end     
      elseif R(j)<1
        % retreat on sg scale
        h_sg = par.Hcr*R(j);
        H = h_sg - abs(dh(j));
        if H>0
          dhdt_melt(j) = dhdt(j);
          % partial sg melt
          R(j) = find_R(H,par);
          h(j) = 0; 
          Ts(j) = par.T_f;
        elseif H<=0
          dhdt_melt(j) = -h_sg/par.dt;
          % sg melts completely, warming the ocean potentially
          h(j) = 0;
          R(j) = 0;
          Ts(j) = NaN; 
          dE_used = dhdt_melt(j)*par.rho_i*par.L_f;
          dE_rem = abs(dE(j))-abs(dE_used);
          if dE_rem<0
            fprintf(1,'error_1:dE_rem<0');
            keyboard;
          end
          To(j) = To(j) + par.dt*(par.rho_o*par.c_o*par.h_o)^(-1)*dE_rem;
        end
      end
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Ts,dhdt_melt]=Ts_check(h,par,qa,R,Ta,To,Ts,n)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Checks Ts for Unphysical Values and sets ice to Tf when Ts>Tf
% It is "assumed" that surface ice melts when doing so, but this melt
% is not explicitly calculated.
dhdt = zeros(par.nj,1);
dh = zeros(par.nj,1);
dhdt_melt = zeros(par.nj,1);

Tsi = find(Ts>par.T_f);

if length(Tsi)>0

  Ts2 = ones(par.nj,1)*par.T_f;

  C = Cflux(h,R,To,Ts,par);
  Qs = Q_s(R,par);
  Is = I_s(R,Ta,Ts,par);
  Hs = H_s(R,Ta,Ts,par);
  Es=E(R,Ta,To,Ts,qa,par);

  C2 = Cflux(h,R,To,Ts2,par);
  Qs2 = Q_s(R,par);
  Is2 = I_s(R,Ta,Ts2,par);
  Hs2 = H_s(R,Ta,Ts2,par);
  Es2=E(R,Ta,To,Ts2,qa,par);

  E = Qs + Is + Hs - par.L_v*Es + C; 
  E2 = Qs2 + Is2 + Hs2 - par.L_v*Es2 + C2; 

  dE = E - E2; 
  dhdt = dE/(par.rho_i*par.L_f);
  % Reset Ts
  Ts(Tsi) = par.T_f;
  dhdt_melt(Tsi) = dhdt(Tsi);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h,R,To,Ts,dhdt_frz]=To_check(h,n,par,R,To,Ts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Checks To everywhere and grows ice where To<Tf
% Note: Function no longer melts ice where To>Tf bc diffusion of heat
% under thick ice is critical mechanism in obtaining physical sea ice thicknesses

dhdt = zeros(par.nj,1);
dhdt_frz = zeros(par.nj,1);
dhdt_melt = zeros(par.nj,1);
dh1 = zeros(par.nj,1);
dh2 = zeros(par.nj,1);

Toi_frz = find(To<par.T_f);
if length(Toi_frz)>0

  for i = 1:length(Toi_frz)
    j = Toi_frz(i); 

    dhdt(j) = par.rho_o*par.h_o*par.c_o*(par.T_f - To(j))/(par.rho_i*par.L_f*par.dt);
    dh1(j) = dhdt(j)*par.dt;
    
    % Accounting for Ice Frozen in Surface E Balance
    dhdt_frz(j) = dhdt(j); 

    if dh1(j)>=0 
      if R(j,3)==1
        % Ts stays the same
        H = h(j) + dh1(j);
        % Advance on Grid Scale
        h(j) = H;
        R(j,3) = 1;
      elseif R(j,3)==0
        H = dh1(j);
        % dhdt_frz(j) = dhdt(j); %surface frz in surface balance
        if H>=par.Hcr
          Ts(j) = Ts_extrap_EBM(j,R,Ts,par);
          % advance into grid scale
          h(j) = H;
          R(j,3) = 1;
        else
          Ts(j) = Ts_extrap_EBM(j,R,Ts,par);
          % advance on sg scale
          h(j) = 0;
          R(j,3) = find_R(H,par);
        end
      elseif R(j,3)>0 & R(j,3)<1 
        H = R(j,3)*par.Hcr + dh1(j);
       % dhdt_frz(j) = dhdt(j); %surface frz in surface balance
        if H>=par.Hcr
          % Ts stays the same
          % advance into grid scale
          h(j) = H;
          R(j,3) = find_R(h(j),par);
        else 
          % Ts stays the same
          % advance on sg scale
          h(j) = 0;
          R(j,3) = find_R(H,par);
        end
      else
        fprintf(1,'error: R<0\n');
        keyboard;
      end
    else
      fprintf(1,'error: Toi_frz, dh1<0\n');
      keyboard;
    end
  end
  To(Toi_frz)=par.T_f; % output new To
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h,R,To,Ts,dhdt_frz]=To_check_old(h,n,par,R,To,Ts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Checks To everywhere and grows ice where To<Tf
% Note: Function no longer melts ice where To>Tf bc diffusion of heat
% under thick ice is critical mechanism in obtaining physical sea ice

dhdt = zeros(par.nj,1);
dhdt_frz = zeros(par.nj,1);
dhdt_melt = zeros(par.nj,1);
dh1 = zeros(par.nj,1);
dh2 = zeros(par.nj,1);

Toi_frz = find(To<par.T_f);
if length(Toi_frz)>0

  for i = 1:length(Toi_frz)
    j = Toi_frz(i); 

    dhdt(j) = par.rho_o*par.h_o*par.c_o*(par.T_f - To(j))/(par.rho_i*par.L_f*par.dt);
    dh1(j) = dhdt(j)*par.dt;
    
    % Accounting for Ice Frozen in Surface E Balance
    dhdt_frz(j) = dhdt(j); 

    if dh1(j)>=0 
      if R(j)==1
        % Ts stays the same
        H = h(j) + dh1(j);
        % Advance on Grid Scale
        h(j) = H;
        R(j) = 1;
      elseif R(j)==0
        H = dh1(j);
        % dhdt_frz(j) = dhdt(j); %surface frz in surface balance
        if H>=par.Hcr
          Ts(j) = par.T_f;
          % advance into grid scale
          h(j) = H;
          R(j) = 1;
        else
          Ts(j) = par.T_f;
          % advance on sg scale
          h(j) = 0;
          R(j) = find_R(H,par);
        end
      elseif R(j)>0 & R(j)<1 
        H = R(j)*par.Hcr + dh1(j);
       % dhdt_frz(j) = dhdt(j); %surface frz in surface balance
        if H>=par.Hcr
          % Ts stays the same
          % advance into grid scale
          h(j) = H;
          R(j) = find_R(h(j),par);
        else 
          % Ts stays the same
          % advance on sg scale
          h(j) = 0;
          R(j) = find_R(H,par);
        end
      else
        fprintf(1,'error: R<0\n');
        keyboard;
      end
    else
      fprintf(1,'error: Toi_frz, dh1<0\n');
      keyboard;
    end
  end
  To(Toi_frz)=par.T_f; % output new To
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gm = globmean(f,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Take the global area-weighted mean
x = zeros(par.nj-1,1);
area = zeros(par.nj-1,1);
for j=1:(par.nj-1)
 x(j) = (f(j+1)+f(j))*(par.c(j+1)-par.c(j))/2;
 area(j) = (par.c(j+1)-par.c(j));
end
gm = sum(x)/sum(area); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function lm = latmean(f,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Take the area-weighted mean of the positive latitudinal section 
% of diffusive terms to calculate the poleward heat transport. 
x = zeros(par.EQ-1,1);
area = zeros(par.EQ-1,1);
index = find(f(1:(par.EQ-1))>0);
for i = 1:length(index)
  j = index(i);
  x(j) = (f(j+1)+f(j))*(par.c(j+1)-par.c(j))/2;
  area(j) = (par.c(j+1)-par.c(j));
end
lm = sum(x)/sum(area); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function int = specint(f,phi,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Take the integral from the NP up to a given colatitude
% This function is used in calculating the heat transport from E imbalances.
% phi is the index of the colatitude
x = zeros(phi,1);
for j=1:phi
  x(j) = (f(j+1)+f(j));
end
% check that par.theta is in latitude, not colatitude - xx
int = ((par.theta(phi)-par.theta(1))*pi/180)*sum(x)/(2*phi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [atm_E,ocn_E,netatm_E,netocn_E,atm_diff,ocn_diff,netatm_diff,netocn_diff] = hflux(par,Surf,ToA,diff_To_np1,diff_Ta_np1,diff_qa_np1); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates the total ocean and atmospheric heat transport
% as a function of latitude. 
atm = zeros(par.nj,1);
ocn = zeros(par.nj,1);
area = zeros(par.nj,1);

% Heat Transport (based on energy imbalances)
f = -(ToA - Surf).*par.s';
g = -Surf.*par.s';
for j = 1:(par.nj-1)
  phi = j;
  atm(j) = specint(f,phi,par);
  ocn(j) = specint(g,phi,par);   
end
mean_atm = specint(f,par.nj-1,par);
mean_ocn = specint(g,par.nj-1,par);

atm_E = (2*pi*par.R^2*(atm-mean_atm))/1e15; % output in PW
ocn_E= (2*pi*par.R^2*(ocn-mean_ocn))/1e15; % output in PW

netatm_E = specint(atm_E,par.EQ-1,par);
netocn_E = specint(ocn_E,par.EQ-1,par); 

%keyboard;

% Heat Transport (based on diffusion terms)
%atm_diff = 2*pi*par.R^2*diff_Ta_np1/(1e15);
%ocn_diff = 2*pi*par.R^2*diff_To_np1/(1e15);

%netatm_diff = 2*pi*par.R^2*latmean(diff_Ta_np1,par)/(1e15);
%netocn_diff = 2*pi*par.R^2*latmean(diff_To_np1,par)/(1e15);

atm2 = zeros(par.nj,1);
ocn2 = zeros(par.nj,1);
area2 = zeros(par.nj,1);

% Heat Transport (based on diffusion terms)
f2 = diff_Ta_np1.*par.s' + par.L_v*diff_qa_np1.*par.s';
g2 = diff_To_np1.*par.s';
for j = 1:(par.nj-1)
  phi = j;
  atm2(j) = specint(f2,phi,par);
  ocn2(j) = specint(g2,phi,par);
end
mean_atm2 = specint(f2,par.nj-1,par); 
mean_ocn2 = specint(g2,par.nj-1,par);

atm_diff = 2*pi*par.R^2*(atm2-mean_atm2)/1e15;
ocn_diff = 2*pi*par.R^2*(ocn2-mean_ocn2)/1e15;

netatm_diff = 2*pi*par.R^2*(specint(atm_diff,par.EQ-1,par));
netocn_diff = 2*pi*par.R^2*(specint(ocn_diff,par.EQ-1,par));

%keyboard;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function Fi_np1 = EBM_F_i(R,Fi_initial,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function adjusts the long-term horizontal convergences
% with respect to the initial and current ice fields

%% This function uses a rather inaccurate method to toggle 
%% long term convergences on or off - xx

%Fi_np1 = zeros(par.nj,1);

%for j = 2:(par.nj-1)
  %if j<par.EQ
    %if R(j-1)<1
      %Fi_np1(j) = 0;   
    %elseif R(j-1)==1 
      %Fi_np1(j) = Fi_initial(j); 
    %end 
  %elseif j>par.EQ
    %if R(j+1)<1
      %Fi_np1(j) = 0;
    %elseif R(j+1)==1
      %Fi_np1(j) = Fi_initial(j);
    %end 
  %else
    %Fi_np1(j) = 0;
  %end
%end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [par] = EBM_iceline(h,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function determines the ice line and ends EBM if:
% (1) ice retreats to the poles
% (5) EBM reaches energy balance before 5 year upper limit
%     and after lower limit
%% Operation includes
% i) final round of plots
% ii) make restart file
% iii) FIS early termination


%lowerlim = par.year/par.dt;
%upperlim = par.nt;

% Nedge Update
%arg_NH = min(find(h(1:par.EQ,3)==0));
%if length(arg_NH)~=0
  %Nedge = arg_NH-1;
%else
  %Nedge = 0;
%end
% Sedge Update
%arg_SH = max(find(h(par.EQ:end,3)==0));
%if length(arg_SH)~=0
  %Sedge = arg_SH+1+44;
%else
  %Sedge = 0;
%end

% Operation (1)
%if par.EBM_expnum==1
  %if Nedge==1 | Sedge==1
    %par.icelatpole = 1;
    %fprintf(1,sprintf('EBMpar.icelatpole = 1, EBM_expnum %.2d, n = %.2d\n',par.EBM_expnum,par.n));
  %end
%end

%% Operation (5)
% This operation could be an issue if we do not let the ocean equilibrate -XX
%if abs(netSurf)<0.5 && abs(netToA)<0.5 && par.n>=ceil(lowerlim) && par.n<upperlim
  %par.ebalance = 1; 
  % flag run for early termination
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Hcr = update_Hcr2(h,R,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Declare Maximal Hcr - XX arbitrary
Hcr_max = 20;
Hcr_min = 0.001;

if par.Hcr<Hcr_max 
  if sum(h)==0
    Hcr = min(R)*par.Hcr;
    if Hcr<Hcr_min
      Hcr = Hcr_min;
    end
  elseif min(h)>=Hcr_min
    Hcr = min(h);
  else
    Hcr = Hcr_min;
  end
else
  % no change
  Hcr = Hcr_max; 
end 

if Hcr<=0 | Hcr>Hcr_max
  fprintf(1,'error: unphysical Hcr\n');
  keyboard;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Hcr = update_Hcr(h,R,par)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Hcr = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function par = integrate_EBM_1d_sphere(par,var2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ice flow in 1 horizontal dimension using spherical coordinates.

% Create Folder to Export Plots
plotfolder_EBM='/u/scratch/f/fspauldi/exosnowballs/EBMFigures';

plotfolder = sprintf('%s/EBM-nonlinear-exp-%.2d-Q-%.2d-N-%.2d-eps-%.2d-%s/',plotfolder_EBM,par.EBM_expnum,par.Qo,par.N,100*par.A_a,var2);
%if par.N==500
if ~exist(sprintf('%s',plotfolder),'file') & (par.N==1 || floor(par.N/par.Nplot)*par.Nplot==par.N || par.N==par.Nt)
  [status,msg] = mkdir(sprintf('%s',plotfolder));
elseif exist(sprintf('%s',plotfolder),'file') & (par.N==1 || floor(par.N/par.Nplot)*par.Nplot==par.N || par.N==par.Nt)
  [status,msg] = rmdir(sprintf('%s',plotfolder),'s');
  [status,msg] = mkdir(sprintf('%s',plotfolder));
end 
%end

% Create Folder to Export EBM Restarts (i.e. FIS output)
restartfolder_EBM = '/u/scratch/f/fspauldi/exosnowballs/EBMRestart';
if ~exist(sprintf('%s',restartfolder_EBM),'file')
  [status,msg] = mkdir(sprintf('%s',restartfolder_EBM));
end

%% initialize variables:
Ta = zeros(par.nj,3);
To = zeros(par.nj,3);
qa = zeros(par.nj,3);
Ts = NaN(par.nj,3);

M1 = zeros(par.nj,1);
M2 = zeros(par.nj,1);
M3 = zeros(par.nj,1);

Qs_n = zeros(par.nj,1);

P_n = zeros(par.nj,1);
E_n = zeros(par.nj,1);

P_np1 = zeros(par.nj,1);
E_np1 = zeros(par.nj,1);

Qo_n = zeros(par.nj,1);
Io_n = zeros(par.nj,1);
Ho_n = zeros(par.nj,1);
G_n = zeros(par.nj,1);

Qo_np1 = zeros(par.nj,1);
Io_np1 = zeros(par.nj,1);
Ho_np1 = zeros(par.nj,1);    
G_np1 = zeros(par.nj,1);

Is_n = zeros(par.nj,1);
Hs_n = zeros(par.nj,1);
Is_np1 = zeros(par.nj,1);
Hs_np1 = zeros(par.nj,1);

Qa_n = zeros(par.nj,1);
Ia_n = zeros(par.nj,1);
Ha_n = zeros(par.nj,1);

Qa_np1 = zeros(par.nj,1);
Ia_np1 = zeros(par.nj,1);
Ha_np1 = zeros(par.nj,1);

rhoa_n = zeros(par.nj,1);
rhoa_np1 = zeros(par.nj,1);

diff_To_n = zeros(par.nj,1);
diff_Ta_n = zeros(par.nj,1);
diff_qa_n = zeros(par.nj,1);

diff_To_np1 = zeros(par.nj,1);
diff_Ta_np1 = zeros(par.nj,1);
diff_qa_np1 = zeros(par.nj,1);

L_n = zeros(par.nj,1);
L_np1 = zeros(par.nj,1);

rec = NaN(par.nt,7);

Fo_n = zeros(par.nj,1);
C_n = zeros(par.nj,1);
Pi_n = zeros(par.nj,1);

Fo_np1 = zeros(par.nj,1);
C_np1 = zeros(par.nj,1);
Pi_np1 = zeros(par.nj,1);

dhdt_frz = zeros(par.nj,1);
dhdt_melt = zeros(par.nj,1);

% Load Model Variables
[h,R,Ta,To,Ts,qa]=EBM_read_restart(par,var2);
fprintf(1,'running 1D spherical EBM @ N = %.2d, Q=%.2d, EBM_expnum=%.2d, epsilon =%.2d, ice %s\n',par.N,par.Qo,par.EBM_expnum,100*par.A_a,var2);

par.h_initial = h(:,2);
par.R_initial = R(:,2);
par.To_initial = To(:,2);
par.To_init_ave = globmean(To(:,2),par);

% Load n=1 Model Fields

Qs_n = Q_s(R(:,1),par); 

P_n=P(R(:,1),Ta(:,1),Ts(:,1),qa(:,1),par,plotfolder,1);
E_n=E(R(:,1),Ta(:,1),To(:,1),Ts(:,1),qa(:,1),par);

Qo_n = Q_o(R(:,1),par);
Io_n = I_o(R(:,1),Ta(:,1),To(:,1),par);
Ho_n = H_o(R(:,1),Ta(:,1),To(:,1),par);    
G_n(:) = par.G;

Is_n = I_s(R(:,1),Ta(:,1),Ts(:,1),par); 
Hs_n = H_s(R(:,1),Ta(:,1),Ts(:,1),par); 

Qa_n = Q_a(R(:,1),par);
Ia_n = I_a(R(:,1),par,Ta(:,1),To(:,1),Ts(:,1),1);
Ha_n = H_a(R(:,1),Ho_n,Hs_n,par);    

rhoa_n = rho_a(Ta(:,1),par);

C_n = Cflux(h(:,1),R(:,1),To(:,1),Ts(:,1),par);
Fo_n = F_o(C_n,G_n,par,R);
Pi_n = P_i(P_n,Ts(:,1),par);

L_n(:) = par.L_v;

track_melt = NaN(length(par.nt),1);

print = 1;
%% Time step equation:
for n=1:par.nt  

  if n==print
    EBM_n = n
    print = print + 18143;
  end

  % Update time step
  par.n=n;
  time_kyr=n*par.dt/par.year/1000;

  %% Advance qa, Ta, To, Ts in time using 2nd Order Adams Bashforth

  % Compute Model fields of current time step

  Qs_np1 = Q_s(R(:,2),par);

  P_np1=P(R(:,2),Ta(:,2),Ts(:,2),qa(:,2),par,plotfolder,n);

  E_np1=E(R(:,2),Ta(:,2),To(:,2),Ts(:,2),qa(:,2),par);

  Qo_np1 = Q_o(R(:,2),par);
  Io_np1 = I_o(R(:,2),Ta(:,2),To(:,2),par);
  Ho_np1 = H_o(R(:,2),Ta(:,2),To(:,2),par);    
  G_np1(:) = par.G;

  Is_np1 = I_s(R(:,2),Ta(:,2),Ts(:,2),par); 
  Hs_np1 = H_s(R(:,2),Ta(:,2),Ts(:,2),par); 

  Qa_np1 = Q_a(R(:,2),par);
  Ia_np1 = I_a(R(:,2),par,Ta(:,2),To(:,2),Ts(:,2),n);
  Ha_np1 = H_a(R(:,2),Ho_np1,Hs_np1,par);

  rhoa_np1 = rho_a(Ta(:,2),par);

  C_np1 = Cflux(h(:,2),R(:,2),To(:,2),Ts(:,2),par);
  Fo_np1 = F_o(C_np1,G_np1,par,R(:,2));
  Pi_np1 = P_i(P_np1,Ts(:,2),par);

  L_np1(:) = par.L_v;

  if sum(isnan(Ts(:,3)))~=par.nj
    fprintf(1,'error: leading condition of Ts loop is wrong\n');
    keyboard;
  end

  for j = 2:(par.nj-1)

    s_jphalf = 0.5*(par.s(j)+par.s(j+1)); 
    s_jmhalf = 0.5*(par.s(j)+par.s(j-1));

    %% Atmospheric Specific Humidity, qa

    diff_qa_n(j) = (par.rho_a*par.h_q*par.D_q/(par.R^2*par.s(j)))*(1/par.dtheta_rad^2)*(s_jphalf*(qa(j+1,1)-qa(j,1)) - s_jmhalf*(qa(j,1)-qa(j-1,1)));
 
    RHS_n = (par.rho_a*par.h_q)^(-1)*(diff_qa_n(j) - (P_n(j) - E_n(j)));
    
    diff_qa_np1(j) = (par.rho_a*par.h_q*par.D_q/(par.R^2*par.s(j)))*(1/par.dtheta_rad^2)*(s_jphalf*(qa(j+1,2)-qa(j,2)) - s_jmhalf*(qa(j,2)-qa(j-1,2))); 

    RHS_np1 = (par.rho_a*par.h_q)^(-1)*(diff_qa_np1(j) - (P_np1(j) - E_np1(j)));

    qa(j,3) = qa(j,2) + par.dt*(1.5*RHS_np1-0.5*RHS_n); 


    %% Ocean Mixed-Layer Temperature, To
    % note: ocean diffusion is zero under complete ice cover (sg weighted)
    %odiff_n = par.D_o*(1-R(j,1));
    %odiff_np1 = par.D_o*(1-R(j,2));  
    odiff_n = par.D_o;
    odiff_np1 = par.D_o;

    diff_To_n(j) = (par.rho_o*par.c_o*par.h_o*odiff_n/(par.R^2*par.s(j)))*(1/par.dtheta_rad^2)*(s_jphalf*(To(j+1,1)-To(j,1)) - s_jmhalf*(To(j,1)-To(j-1,1))); 

    RHS_n = (par.rho_o*par.c_o*par.h_o)^(-1)*(diff_To_n(j) + (1-R(j,1))*Qo_n(j) + (1-R(j,1))*Io_n(j) + (1-R(j,1))*Ho_n(j) - par.L_v*E_n(j)*(1-R(j,1)) + G_n(j) - R(j,1)*par.Beta*(To(j,1)-par.T_f));

    diff_To_np1(j) = (par.rho_o*par.c_o*par.h_o*odiff_np1/(par.R^2*par.s(j)))*(1/par.dtheta_rad^2)*(s_jphalf*(To(j+1,2)-To(j,2)) - s_jmhalf*(To(j,2)-To(j-1,2)));

    RHS_np1 = (par.rho_o*par.c_o*par.h_o)^(-1)*(diff_To_np1(j) + (1-R(j,2))*Qo_np1(j) + (1-R(j,2))*Io_np1(j) + (1-R(j,2))*Ho_np1(j) - par.L_v*E_np1(j)*(1-R(j,2)) + G_np1(j) - R(j,2)*par.Beta*(To(j,2)-par.T_f));
   
    To(j,3) = To(j,2) + par.dt*(1.5*RHS_np1-0.5*RHS_n); 

    %% Atmospheric near-surface air temperature, Ta   
  
   diff_Ta_n(j) = (par.rho_a*par.c_a*par.h_a*par.D_a/(par.R^2*par.s(j)))*(1/par.dtheta_rad^2)*(s_jphalf*(Ta(j+1,1)-Ta(j,1)) - s_jmhalf*(Ta(j,1)-Ta(j-1,1)));

    RHS_n = (par.rho_a*par.c_a*par.h_a)^(-1)*(diff_Ta_n(j) + Qa_n(j) + Ia_n(j) + Ha_n(j) + par.L_v*P_n(j));

    diff_Ta_np1(j) = (par.rho_a*par.c_a*par.h_a*par.D_a/(par.R^2*par.s(j)))*(1/par.dtheta_rad^2)*(s_jphalf*(Ta(j+1,2)-Ta(j,2)) - s_jmhalf*(Ta(j,2)-Ta(j-1,2)));

    RHS_np1 = (par.rho_a*par.c_a*par.h_a)^(-1)*(diff_Ta_np1(j) + Qa_np1(j) + Ia_np1(j) + Ha_np1(j) + par.L_v*P_np1(j));
   
    Ta(j,3) = Ta(j,2) + par.dt*(1.5*RHS_np1-0.5*RHS_n);
   
    % Ice Surface Temperature, Ts

    if R(j,1)>0 & R(j,2)>0
      % if both RHS_n & RHS_np1 refer to ice cell
      RHS_n = (par.rho_i*par.c_i*par.d_i)^(-1)*(Qs_n(j)+Is_n(j)+Hs_n(j)-par.L_v*E_n(j)+C_n(j)); %L_s
      RHS_np1 = (par.rho_i*par.c_i*par.d_i)^(-1)*(Qs_np1(j)+Is_np1(j)+Hs_np1(j)-par.L_v*E_np1(j)+C_np1(j)); %L_s
      Ts(j,3) = Ts(j,2) + par.dt*(1.5*RHS_np1-0.5*RHS_n);

    elseif R(j,1)==0 & R(j,2)>0
      % if RHS_n corresponds to open ocean cell and RHS_np1 corresponds to ice cell
      RHS_np1 = (par.rho_i*par.c_i*par.d_i)^(-1)*(Qs_np1(j)+Is_np1(j)+Hs_np1(j)-par.L_v*E_np1(j)+C_np1(j)); %L_s
      Ts(j,3) = Ts(j,2) + par.dt*RHS_np1;

    elseif R(j,1)>0 & R(j,2)==0
      % if RHS_n corresponds to an ice cell and RHS_np1 corresponds to ocean cell
      Ts(j,3) = NaN;
    elseif R(j,1)==0 & R(j,2)==0
      % if both RHS_n & RHS_np1 refer to ocean cell
      Ts(j,3) = NaN; 
    else
      fprintf(1,'error: Ts prognostic equation\n');
      keyboard;
    end

    % Ice Thickness Equation
    % Pi_extra can only increase the ice thickness
    %if Pi_np1(j)==0
      %h(j,3) = h(j,2);
      %R(j,3) = R(j,2);
    %elseif Pi_np1(j)>0
      %dhdt_extra = Pi_np1(j)/par.rho_i;
      %dh_extra = dhdt_extra*par.dt;
      %if dh_extra<0
        %fprintf(1,'error: dh extra < 0');
        %keyboard;
      %end
      %if R(j,2)==1
        %h(j,3) = h(j,2) + dh_extra;
        %R(j,3) = find_R(h(j,3),par); 
      %elseif R(j,2)==0
        %fprintf(1,'error: there shouldnt be open ocean in ice thickness eq\n');
        %keyboard;
      %else
        % subgrid
        %h_np1 = R(j,2)*par.Hcr; 
        %h(j,3) = h_np1 + dh_extra;
        %if h(j,3)<par.Hcr && h(j,3)>0
          %R(j,3) = find_R(h(j,3),par);
          %h(j,3) = 0;
        %elseif h(j,3)>=par.Hcr
          % h(j,3) is as defined above. 
          %R(j,3) = find_R(h(j,3),par);
        %end
      %end      
    %end

    %% New Ice Thickness Equation
    h(j,3) = h(j,2);
    R(j,3) = R(j,2);

  end
  %keyboard; 

  % set north and south boundary conditions
  Ta(1,3) = Ta(2,3);
  To(1,3) = To(2,3);
  qa(1,3) = qa(2,3); 
  Ts(1,3) = Ts(2,3);

  Ta(par.nj,3) = Ta(par.nj-1,3);
  To(par.nj,3) = To(par.nj-1,3);
  qa(par.nj,3) = qa(par.nj-1,3);
  Ts(par.nj,3) = Ts(par.nj-1,3);

  h(1,3) = h(2,3);
  h(par.nj,3) = h(par.nj-1,3);
  R(1,3) = R(2,3);
  R(par.nj,3) = R(par.nj-1,3);

  if sum(isnan(Ta(:,3)))>0 | sum(isnan(Ia_n))>0 | sum(isnan(Ha_n))>0
    fprintf(1,'Check for NaNs\n');
    keyboard;
  end

  % Check for Unphysical Values of Ts & Melt Surface Layer if Ts>Tf
  % melt water assumed to drain to ocean @ freezing temp
  [Ts(:,3),dhdt_melt]=Ts_check(h(:,3),par,qa(:,3),R(:,3),Ta(:,3),To(:,3),Ts(:,3),n);

  track_melt(n) = globmean(dhdt_melt,par);

  % Check for Unphysical Ocean Temperatures
  % Once the EBM has equilibrated, freeze some ice if To<Tf or melt some ice (if it exists) where To>Tf

  if n==par.nt
  	[h(:,3),R,To(:,3),Ts(:,3),dhdt_frz]=To_check(h(:,3),n,par,R,To(:,3),Ts(:,3));
  end

  % Check Ocean Temperatures under Thick Ice
  %Toi = find(R(:,3)==1 & ((To(:,3)~=par.T_f));
  %Toi_len = length(Toi);
  %if Toi_len>0
    %fprintf(1,'error: To~=Tf under ice\n');
    %keyboard;
  %end
  
  % Constrain Ocean Temperatures under Ice
  %Toi = find(abs(To(:,3)-par.T_f)>2 & isnan(Ts(:,3))==0); 
  %Toi_len = length(Toi);
  %if Toi_len>0
    %fprintf(1,'error: abs(To-Tf)>2 under ice\n');
    %keyboard;
  %end

  % Check for Unphysical Ocean Temperatures
  Toi = find(To(:,3)>373.15);
  Toi_len = length(Toi);
  if Toi_len>0
    fprintf(1,'error: To > Tboil\n');
    keyboard;
  end

  if sum(isnan(R(:,3)))>0 | length(find(h(:,3)==10000))>0
    fprintf(1,'Check for NaNs in R or h = 1e4, i.e. resets\n');
    keyboard;
  end

  % Check ToA and Surface Energy Balance
  if n==1
    Toplot = globmean(To(:,2),par);
    Tsplot = nansum(Ts(:,2))/sum(isnan(Ts(:,2))==0);
    Eplot = zeros(1,2);
  end
  %if n==par.nt
  [Eplot_n,netPrec,netEvap,Aplot,Surf,ToA] = ebalance(Ha_np1,par,R(:,2),Ta(:,2),To(:,2),Ts(:,2),dhdt_frz,dhdt_melt,n,C_np1,G_np1,P_np1,E_np1,rec,var2,Qo_np1,Qs_np1,qa(:,2));
  %end
  Toplot = horzcat(Toplot,globmean(To(:,2),par));
  Tsplot = horzcat(Tsplot,nansum(Ts(:,2))/sum(isnan(Ts(:,2))==0));
  Eplot = vertcat(Eplot,Eplot_n);

  if n==par.nt
    par.Eplot_toa = Eplot(end,1);
    par.Eplot_sfc = Eplot(end,2);
    par.netPrec = netPrec;
    par.netEvap = netEvap;
    par.Aplot = Aplot;
    %keyboard;
  end

  st = ceil(par.nt/2);
  %st = 1;
  tt = (st:par.nt)/par.year;

  if (par.N==1 || floor(par.N/par.Nplot)*par.Nplot==par.N || par.N==par.Nt) & (n==par.nt)
  %if n==par.nt
    clf; 
    fig = figure(1);
    set(fig,'Visible','off');
    subplot(4,1,1)
    plot(tt,Toplot(st+1:end));
    xl = xlabel('Time');
    yl = ylabel('Ocean Temp (K)');
    tl = title(sprintf('Toplot = %.2d',Toplot(end)));
    set([gca,xl,yl,tl],'fontsize',10);

    subplot(4,1,2)
    plot(tt,Tsplot(st+1:end));
    xl = xlabel('Time');
    yl = ylabel('Ice Temp (K)');
    tl = title(sprintf('Tsplot = %.2d',Tsplot(end)));
    set([gca,xl,yl,tl],'fontsize',10);

    subplot(4,1,3)
    plot(tt,Eplot(st+1:end,1));
    xl = xlabel('Time');
    yl = ylabel('Imbalance (W/m^2)');
    tl = title(sprintf('ToA Eplot=%.2d',Eplot(end,1)));
    set([gca,xl,yl,tl],'fontsize',10);

    subplot(4,1,4)
    plot(tt,Eplot(st+1:end,2));
    xl = xlabel('Time');
    yl = ylabel(' Imbalance (W/m^2)');
    tl = title(sprintf('Surf Eplot=%.2d',Eplot(end,2)));
    set([gca,xl,yl,tl],'fontsize',10);

    saveas(gcf,sprintf('%s/eBalance_time.png',plotfolder)); 
  end

  %% -----------
  %% plot Ta,To,qa,h,R
  %% -----------
  if n==1 || floor(n/par.nplot)*par.nplot==n || n==par.nt || par.icelatlim==1 || par.icelatpole==1 || par.ebalance==1

    if (par.N==1 || floor(par.N/par.Nplot)*par.Nplot==par.N || par.N==par.Nt) & (n==par.nt || par.icelatlim==1 || par.icelatpole==1 || par.ebalance==1)
      jb = 2; je = par.nj-1;

       prec_np1 = P_np1*par.year*100/par.rho_o;
       evap_np1 = E_np1*par.year*100/par.rho_o;

       % Figure 1: Precipitation
       fig = figure(1); clf
       set(fig,'Visible','off');
       plot(90-par.theta(je:-1:jb),prec_np1(je:-1:jb),'--ok');
       yl = ylabel('Precipitation (cm/yr)');
       xl = xlabel('Latitude');
       tl = title('Precipitation @ t=np1');
       set([gca,xl,yl,tl],'fontsize',10); 

       saveas(gcf,sprintf('%s/Px-%.2d.png',plotfolder,n)); 

       % Figure 2: Evaporation
       fig = figure(2); clf
       set(fig,'Visible','off');
       plot(90-par.theta(je:-1:jb),evap_np1(je:-1:jb));
       yl = ylabel('Evaporation (cm/yr)');
       xl = xlabel('Latitude');
       tl = title('Evaporation @ t=n');
       set([gca,xl,yl,tl],'fontsize',10); 

       saveas(gcf,sprintf('%s/Ex-%.2d.png',plotfolder,n)); 

       % Figure 3: Net Solar Radiative Fluxes Absorbed by Surface 
       fig = figure(3); clf
       set(fig,'Visible','off');
       plot(90-par.theta(je:-1:jb),Qo_np1(je:-1:jb),'--ob',90-par.theta(je:-1:jb),Qs_np1(je:-1:jb),'--or',90-par.theta(je:-1:jb),Qa_np1(je:-1:jb),'--ok');
       yl = ylabel('W/m^2');
       xl = xlabel('Latitude');
       tl = title('Net Solar Radiative Flux Absorbed by Surface @ t=np1');
       legend Qo_np1 Qs_np1 Qa_np1
       set([gca,xl,yl,tl],'fontsize',10);

       saveas(gcf,sprintf('%s/Qx-%.2d.png',plotfolder,n)); 

       % Figure 4: Net Infrared Flux Absorbed by Surface
       fig = figure(4); clf
       set(fig,'Visible','off');
       plot(90-par.theta(je:-1:jb),Io_np1(je:-1:jb),'--ob',90-par.theta(je:-1:jb),Is_np1(je:-1:jb),'--or',90-par.theta(je:-1:jb),Ia_np1(je:-1:jb),'--ok');
       yl = ylabel('W/m^2');
       xl = xlabel('Latitude');
       tl = title('Net Infrared Flux Absorbed by Surface @ t=np1');
       legend Io_np1 Is_np1 Ia_np1
       set([gca,xl,yl,tl],'fontsize',10);

       saveas(gcf,sprintf('%s/Ix-%.2d.png',plotfolder,n));

       % Figure 5: Sensible Heat Fluxes Transferred to Surface
       fig = figure(5); clf
       set(fig,'Visible','off');
       plot(90-par.theta(je:-1:jb),Ho_np1(je:-1:jb),'--ob',90-par.theta(je:-1:jb),Hs_np1(je:-1:jb),'--or',90-par.theta(je:-1:jb),Ha_np1(je:-1:jb),'--ok');
       yl = ylabel('W/m^2');
       xl = xlabel('Latitude');
       tl = title('Sensible Heat Flux Transferred to Surface @ t=np1');
       legend Ho_np1 Hs_np1 Ha_np1
       set([gca,xl,yl,tl],'fontsize',10);

       saveas(gcf,sprintf('%s/Hx-%.2d.png',plotfolder,n));

       % Figure 6: Surface Temperatures
       numyrs = (n/par.nt)*(par.Time/par.year); % time in yrs

       fig = figure(6); clf
       set(fig,'Visible','off');
       subplot(3,1,1)
       plot(90-par.theta(je:-1:jb),To(je:-1:jb,1)-par.T_f,'--ob',90-par.theta(je:-1:jb),To(je:-1:jb,2)-par.T_f,'--or',90-par.theta(je:-1:jb),To(je:-1:jb,3)-par.T_f,'--ok');
       yl = ylabel('C');
       xl = xlabel('Latitude');
       tl = title(sprintf('Ocean Temperature as a function of latitude at t=%.2d yrs',numyrs));
       set([gca,xl,yl,tl],'fontsize',10); 
       subplot(3,1,2)
       plot(90-par.theta(je:-1:jb),Ts(je:-1:jb,1)-par.T_f,'--ob',90-par.theta(je:-1:jb),Ts(je:-1:jb,2)-par.T_f,'--or');
       yl = ylabel('C');
       xl = xlabel('Latitude');
       tl = title(sprintf('Ice Surface Temperature as a function of latitude at t=%.2d yrs',numyrs));
       set([gca,xl,yl,tl],'fontsize',10);
       subplot(3,1,3)
       plot(90-par.theta(je:-1:jb),Ta(je:-1:jb,1)-par.T_f,'--ob',90-par.theta(je:-1:jb),Ta(je:-1:jb,2)-par.T_f,'--or',90-par.theta(je:-1:jb),Ta(je:-1:jb,3)-par.T_f,'--ok');
       yl = ylabel('C');
       xl = xlabel('Latitude');
       tl = title(sprintf('Atmospheric Temperature as a function of latitude at t=%.2d yrs',numyrs));
       set([gca,xl,yl,tl],'fontsize',10);

       saveas(gcf,sprintf('%s/Tx-%.2d.png',plotfolder,n));

       % Figure 7: Atmospheric Specific Humidity
       fig = figure(7); clf
       set(fig,'Visible','off');
       plot(90-par.theta(je:-1:jb),qa(je:-1:jb,1),'--ob',90-par.theta(je:-1:jb),qa(je:-1:jb,2),'--or',90-par.theta(je:-1:jb),qa(je:-1:jb,3),'--ok');
       yl = ylabel('kg/kg');
       xl = xlabel('Latitude');
       tl = title('Atmospheric Humidity as a function of time');
       set([gca,xl,yl,tl],'fontsize',10); 

       saveas(gcf,sprintf('%s/qa-%.2d.png',plotfolder,n));

       % Figure 8: Fractional Field Height and Sea Glacier Thickness
       fig = figure(8); clf
       set(fig,'Visible','off');
       [ax,hl1,hl2] = plotyy(90-par.theta(je:-1:jb),R(je:-1:jb,2),90-par.theta(je:-1:jb),h(je:-1:jb,2));
       set([hl1,hl2],'Marker','o');
       set([hl1,hl2],'linewidth',2);
       xlim(ax(1),[90-par.theta(je),90-par.theta(jb)]);
       xlim(ax(2),[90-par.theta(je),90-par.theta(jb)]);
       h1 = xlabel(ax(1),'latitude');
       h2a = ylabel(ax(1),'R');
       h2b = ylabel(ax(2),'h (m)');
       h3 = title(sprintf('Fractional Ice Cover (R) and height if uniformly spread over cell, Hcr = %.2d',par.Hcr));
       set([gca,h1,h2a,h2b,h3],'fontsize',10);

       saveas(gcf,sprintf('%s/h-R-%.2d.png',plotfolder,n));
       
       % Figure 9: Diffusion 
       fig = figure(9); clf
       set(fig,'Visible','off');
       subplot(3,1,1)
       plot(90-par.theta(je:-1:jb),diff_To_np1(je:-1:jb),'--ob');
       yl = ylabel('W/m^2');
       xl = xlabel('Latitude');
       tl = title(sprintf('Oceanic Heat Diffusion'));
       set([gca,xl,yl,tl],'fontsize',10); 
       subplot(3,1,2)
       plot(90-par.theta(je:-1:jb),diff_Ta_np1(je:-1:jb),'--ok');
       yl = ylabel('W/m^2');
       xl = xlabel('Latitude');
       tl = title(sprintf('Atmospheric Heat Diffusion'));
       set([gca,xl,yl,tl],'fontsize',10); 
       subplot(3,1,3)
       plot(90-par.theta(je:-1:jb),diff_qa_np1(je:-1:jb),'--ok');
       yl = ylabel('(kg/m^2)/s');
       xl = xlabel('Latitude');
       tl = title('Diffusion of Specific Humidity');
       set([gca,xl,yl,tl],'fontsize',10);

       saveas(gcf,sprintf('%s/diffusion-%.2d.png',plotfolder,n));

       % Figure 10: Ocean and Atmospheric Heat Transport 
       % Total Ocean and Atmospheric Poleward Heat Transport in PW
       [atm_E,ocn_E,netatm_E,netocn_E,atm_diff,ocn_diff,netatm_diff,netocn_diff] = hflux(par,Surf,ToA,diff_To_np1,diff_Ta_np1,diff_qa_np1); 

       fig = figure(10); clf
       set(fig,'Visible','off');
       subplot(2,1,1)
       plot(90-par.theta(je:-1:jb),ocn_E(je:-1:jb),'--b',90-par.theta(je:-1:jb),ocn_diff(je:-1:jb),':r');
       legend ocn-E ocn-diff
       yl = ylabel('PW');
       xl = xlabel('Latitude');
       tl = title(sprintf('Oceanic Heat Flux: ocn-E=%.2d PW, ocn-diff=%.2d PW',globmean(ocn_E,par),globmean(ocn_diff,par)));
       set([gca,xl,yl,tl],'fontsize',10); 
       subplot(2,1,2)
       plot(90-par.theta(je:-1:jb),atm_E(je:-1:jb),'--b',90-par.theta(je:-1:jb),atm_diff(je:-1:jb),':r');
       legend atm-E atm-diff
       yl = ylabel('PW');
       xl = xlabel('Latitude');
       tl = title(sprintf('Atmospheric Heat Flux: atm-E=%.2d PW, atm-diff=%.2d PW',globmean(atm_E,par),globmean(atm_diff,par)));
       set([gca,xl,yl,tl],'fontsize',10); 

       saveas(gcf,sprintf('%s/hflux-%.2d.png',plotfolder,n));
    end

      Hcr = par.Hcr;
      S = (P_np1 - E_np1)/par.rho_i + dhdt_melt; % FIS requires units of m/s 
      % Fo_np1 added to source term in the ice flow model
      P_E = (P_np1-E_np1)/par.rho_i;

    %% Write restart or export - XX
    if (par.N==1 | floor(par.N/par.Nplot)*par.Nplot==par.N || par.N==par.Nt) & (n==par.nt || par.icelatlim==1 || par.icelatpole==1 || par.ebalance==1)

      je=par.nj-1; jb=2;     
      % Figure 10: Source Term
      fig = figure(10); clf
      set(fig,'Visible','off');
      subplot(2,1,1);
      plot(90-par.theta(je:-1:jb),P_E(je:-1:jb)*par.year*100,'--or',90-par.theta(je:-1:jb),Fo_np1(je:-1:jb)*par.year*100,'--ok',90-par.theta(je:-1:jb),dhdt_melt(je:-1:jb)*par.year*100,'--g',90-par.theta(je:-1:jb),dhdt_frz(je:-1:jb)*par.year*100,90-par.theta(je:-1:jb),S(je:-1:jb)*par.year*100,'--ob');
      yl = ylabel('Source Term (cm/yr)');
      xl = xlabel('Latitude');
      tl = title('Source Terms');
      legend P_E Fo dhdt_melt dhdt_frz S
      set([gca,xl,yl,tl],'fontsize',10); 

      subplot(2,1,2);
      plot((par.tfac*n/par.nt)./(n:-1:1),track_melt(1:n)*par.year*100);
      xl = xlabel('time (yrs)');
      yl = ylabel(sprintf('Surface Melt (cm/yr)'));
      tl = title(sprintf('Surface Melt = %2.2d',track_melt(n)*par.year*100));
      set([gca,xl,yl,tl],'fontsize',10); 

      saveas(gcf,sprintf('%s/source-%.2d.png',plotfolder,n));
    end


    % prepare fields for export
    Ta(:,1) = Ta(:,2);
    Ta(:,2) = Ta(:,3);
    To(:,1) = To(:,2);
    To(:,2) = To(:,3); 
    Ts(:,1) = Ts(:,2);
    Ts(:,2) = Ts(:,3);
    qa(:,1) = qa(:,2);
    qa(:,2) = qa(:,3);

    h(:,1)=h(:,2);
    h(:,2)=h(:,3);
    R(:,1)=R(:,2);
    R(:,2)=R(:,3);

    index = find(par.R_initial>0);
    indexlen = length(index);
    hx = zeros(indexlen,1);
    if indexlen>0
      for i = 1:indexlen
	j = index(i);
	if R(j,3)==1
	  hx(j) = par.h_initial(j);
	elseif R(j,3)>0
	  hx(j) = par.R_initial(j)*par.Hcr;
	end
      end 
    end
    % average ice thickness (not area weighted)
    if indexlen>0
      par.h_init_ave = sum(hx)/length(hx);
    else
      par.h_init_ave = 0;
    end

    par.h_min = min(h(find(h(:,3)>0),3));
    par.h_max = max(h(:,3));

    par.h_end = h(:,3);
    par.R_end = R(:,3);

    % mean ocean temp
    par.To_ave = globmean(To(:,3),par);
    % max ocean temp
    par.To_max = max(To(:,3));
    % min ocean temp
    par.To_min = min(To(:,3)); 

    % Average h 
    index = find(R(:,3)>0);
    indexlen = length(index);
    hx = zeros(indexlen,1);
    if indexlen>0
      for i = 1:indexlen
	j = index(i);
	if R(j,3)==1
	  hx(j) = h(j,3);
	elseif R(j,3)>0
	  hx(j) = R(j,3)*par.Hcr;
	end
      end 
    end
    % average ice thickness (not area weighted)
    if indexlen>0
      par.h_ave = sum(hx)/length(hx);
    else
      par.h_ave = 0;
    end

    save(sprintf('%s/restart-EBM-expnum-%.2d-Q-%.2d-resnum-%.2d-eps-%.2d-%s.mat',restartfolder_EBM,par.EBM_expnum,par.Qo,par.N,100*par.A_a,var2),'h','R','Ta','To','Ts','qa','Hcr','S','P_E','diff_Ta_np1','diff_To_np1','diff_qa_np1');

    if par.icelatlim==1 || par.icelatpole==1 || par.ebalance==1
      return;
    end
  end

  %% prepare for next time step:
  Ta(:,1) = Ta(:,2);
  Ta(:,2) = Ta(:,3);
  To(:,1) = To(:,2);
  To(:,2) = To(:,3); 
  Ts(:,1) = Ts(:,2);
  Ts(:,2) = Ts(:,3);
  qa(:,1) = qa(:,2);
  qa(:,2) = qa(:,3);

  Qs_n = Qs_np1;

  P_n = P_np1; 

  E_n = E_np1;

  Qo_n = Qo_np1;
  Io_n = Io_np1;
  Ho_n = Ho_np1;
  G_n = G_np1;

  Is_n = Is_np1;
  Hs_n = Hs_np1;

  Qa_n = Qa_np1;
  Ia_n = Ia_np1;
  Ha_n = Ha_np1;

  rhoa_n = rhoa_np1;

  L_n = L_np1;

  C_n = C_np1;
  Fo_n = Fo_np1;
  Pi_n = Pi_np1;

  h(:,1)=h(:,2);
  h(:,2)=h(:,3);
  R(:,1)=R(:,2);
  R(:,2)=R(:,3);

  % Reset h,R,Ts,To
  h(:,3) = 10000;
  R(:,3) = NaN;
  Ts(:,3) = NaN;
  To(:,3) = 0;
  Ta(:,3) = 10000;
  qa(:,3) = 100;

  % Reset Diagnostic Fields
  dhdt_frz = zeros(par.nj,1);
  dhdt_melt = zeros(par.nj,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function epsilon()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script tests a variety of epsilons, or atmospheric layer 
% absorptivities, par.A_a
% The only change needed for a different variable is the "eps" save text
% and the EBM par.A_a set parameter
eps = [0.85 0.9 0.93 0.96 0.99];
var2 = 'flow';
var3 = 1;

pool = parpool('local',length(eps));

parfor i = 1:length(eps)
  var1 = eps(i);
  FISEBM(var1,var2,var3);
end
fprintf(1,'operation epsilon done.\n');
