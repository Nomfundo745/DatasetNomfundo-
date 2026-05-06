function viz_cyclo_fromUnified(matFile)
% Visualize 15-D cyclostationary features from the merged dataset
% Input MAT (from mmWaveDataset_Balanced_DWPT_fromUnified_ALL.m) must contain:
%   cyclo_f [N,15], labels [N,1], src [N,1] string, fc [N,1], snr [N,1]
%
% Usage:
%   viz_cyclo_fromUnified('mmWaveDataset_Balanced_DWPT_fromUnified_FULL.mat')

clc;
if nargin<1
    matFile = 'mmWaveDataset_Balanced_DWPT_fromUnified_FULL.mat';
    %matFile = 'mmWave_TxRx_Core_SC_FDMA.mat';
end
assert(isfile(matFile),'File not found: %s', matFile);
S = load(matFile);

X   = double(S.cyclo_f);                 % [N,15]
y   = double(S.labels(:));               % 0/1
src = string(S.src(:));                  % source tag
fc  = double(S.fc(:));                   % Hz
snr = double(S.snr(:));                  % dB (NaN for catalog)

% -------- feature names (match your packing order) --------
ebNames = {'EB1_N','EB2_N','EB3_N','EB4_N'};
fn = [ebNames, {'TKEO1','TKEO2','TKEO3','TKEO4', ...
                'Gini1','Gini2','Gini3','Gini4', ...
                'ModDepth1','Entropy','ModIndex'}];

% -------- clean NaNs / infs --------
X(~isfinite(X)) = 0;

% -------- standardize (train-only-style, but we just visualize) --------
mu = mean(X,1,'omitnan'); sg = std(X,0,1,'omitnan'); sg(sg==0)=1;
Z  = (X - mu) ./ sg;

% Colors/markers
cPresent = [0.10 0.55 0.10];  % green
cAbsent  = [0.70 0.10 0.10];  % red
mkSrc    = @(s) pickMarker(s); % function below

%% 1) Per-feature histograms (Absent vs Present)
figure('Color','w','Name','Feature histograms (Absent vs Present)','Position',[50 50 1200 800]);
for i = 1:15
    subplot(3,5,i);
    a = X(y==0,i); a = a(isfinite(a));
    p = X(y==1,i); p = p(isfinite(p));
    histogram(a,30,'FaceAlpha',0.45,'EdgeColor','none','FaceColor',cAbsent); hold on;
    histogram(p,30,'FaceAlpha',0.45,'EdgeColor','none','FaceColor',cPresent);
    title(fn{i}, 'Interpreter','none', 'FontSize',9);
    if i==1, legend('Absent','Present'); legend boxoff; end
end
sgtitle('Cyclostationary feature distributions');

% ----- Dimensionality reduction setup -----
useCols = 1:14;                 % drop ModIndex (column 15)
Z_dr = zscore(Z(:,useCols));    % standardize merged data
X_dr = zscore(X(:,useCols));    % standardize core-only (if you plot it)
fn_dr = fn(useCols);

% ----- PCA (2D) -----
[coeff,score,~,~,expl] = pca(Z_dr,'Algorithm','eig','Centered',true,'NumComponents',3);

figure('Color','w','Name','PCA (2D)','Position',[100 100 900 430]);

% by class
subplot(1,2,1); hold on; grid on; axis equal;
iA = (y==0); iP = (y==1);
scatter(score(iA,1),score(iA,2),16,[0.85 0.40 0.40],'o','filled');   % absent
scatter(score(iP,1),score(iP,2),16,[0.40 0.80 0.55],'x');            % present
xlabel(sprintf('PC1 (%.1f%%)', expl(1))); ylabel(sprintf('PC2 (%.1f%%)', expl(2)));
title('PCA by class'); legend('Absent','Present'); legend boxoff;

% by source + class
subplot(1,2,2); hold on; grid on; axis equal;
srcVals = unique(src);
for k = 1:numel(srcVals)
    idx = src==srcVals(k) & iA; scatter(score(idx,1),score(idx,2),16,[0.85 0.40 0.40],'o','filled');
    idx = src==srcVals(k) & iP; scatter(score(idx,1),score(idx,2),16,[0.40 0.80 0.55],'x');
end
xlabel(sprintf('PC1 (%.1f%%)', expl(1))); ylabel(sprintf('PC2 (%.1f%%)', expl(2)));
title('PCA by source (marker) & class (color)'); 

% ----- t-SNE (or PCA fallback) -----
haveTSNE = exist('tsne','file')==2;
figure('Color','w','Name','t-SNE / PCA fallback','Position',[100 100 520 520]);
if haveTSNE
    try
        Y = tsne(Z_dr,'Perplexity',min(30,round(size(Z_dr,1)/5)),'NumDimensions',2,'Standardize',false);
        ttl = 't-SNE (class colored)';
    catch
        Y = score(:,1:2);
        ttl = 'PCA (fallback)';
    end
else
    Y = score(:,1:2);
    ttl = 'PCA (fallback)';
end
hold on; grid on; axis equal;
scatter(Y(iA,1),Y(iA,2),16,[0.85 0.40 0.40],'o','filled');
scatter(Y(iP,1),Y(iP,2),16,[0.40 0.80 0.55],'x');
title(ttl); legend('Absent','Present'); legend boxoff;


% ----- Correlation heatmap (exclude ModIndex) -----
R = corr(Z(:,useCols),'rows','pairwise');
figure('Color','w','Name','Feature correlation','Position',[50 50 900 750]);
imagesc(R); axis square; colorbar; caxis([-1 1]); colormap(parula);
set(gca,'Fontsize', 12, 'XTick',1:numel(fn_dr),'XTickLabel',fn_dr,'XTickLabelRotation',45, ...
        'YTick',1:numel(fn_dr),'YTickLabel',fn_dr);
title('Correlation matrix (cyclo features, no ModIndex)');

%% 5) Class means +/- sem per feature
m0 = mean(X(y==0,:),1,'omitnan'); m1 = mean(X(y==1,:),1,'omitnan');
s0 = std(X(y==0,:),0,1,'omitnan'); s1 = std(X(y==1,:),0,1,'omitnan');
n0 = max(1,sum(y==0)); n1 = max(1,sum(y==1));
e0 = s0./sqrt(n0); e1 = s1./sqrt(n1);

figure('Color','w','Name','Class means (±SEM)','Position',[80 80 1100 420]);
b = bar([m0; m1].','grouped'); hold on; grid on;
b(1).FaceColor = cAbsent; b(2).FaceColor = cPresent;
% error bars
ng = 2; nb = numel(fn);
xg = nan(ng,nb);
for i = 1:nb
    xg(:,i) = b(1).XEndPoints(i) + [0, (b(2).XEndPoints(i)-b(1).XEndPoints(i))];
end
errorbar(xg(1,:), m0, e0, 'k.', 'LineWidth',1);
errorbar(xg(2,:), m1, e1, 'k.', 'LineWidth',1);
set(gca,'XTick',1:15,'XTickLabel',fn,'XTickLabelRotation',45);
legend('Absent','Present'); title('Per-feature class means ± SEM');

%% 6) Quick radar plot for the 4 normalized energies
% (averaged over class)
radarNames = ebNames;
radarA = m0(1:4); radarP = m1(1:4);
theta = linspace(0,2*pi,5); theta(end) = theta(1);
ra = [radarA, radarA(1)]; rp = [radarP, radarP(1)];

figure('Color','w','Name','Energy bands radar','Position',[80 80 400 400]);
polarplot(theta, ra,'-o','LineWidth',1.5,'Color',cAbsent); hold on;
polarplot(theta, rp,'-o','LineWidth',1.5,'Color',cPresent);
thetaticks(rad2deg(theta(1:4))); thetaticklabels(radarNames);
title('Normalized energy (class averages)'); legend('Absent','Present');

%% 7) Scatter vs SNR or fc (if available)
figure('Color','w','Name','Feature vs SNR/FC','Position',[80 80 1000 420]);
subplot(1,2,1);
fIdx = 14; % Entropy as an example
snrPlot = snr; snrPlot(~isfinite(snrPlot)) = nan; % keep NaNs
scatter_jitter(snrPlot, X(:,fIdx), y, cAbsent, cPresent);
xlabel('SNR (dB, NaN=catalog)'); ylabel(fn{fIdx}); title([fn{fIdx} ' vs SNR']);

subplot(1,2,2);
scatter_jitter(fc/1e9, X(:,fIdx), y, cAbsent, cPresent);
xlabel('Carrier (GHz)'); ylabel(fn{fIdx}); title([fn{fIdx} ' vs fc']);

disp('✅ Visualization complete.');

%% ---------------- helpers ----------------
function mark = pickMarker(s)
    s = string(s);
    if contains(s,'CORE'),      mark = 'o';
    elseif contains(s,'ABS'),   mark = 's';
    elseif contains(s,'Sim','IgnoreCase',true), mark = '^';
    else,                       mark = 'd';
    end
end

function C = colorByClass(y, c0, c1)
    % returns Nx3 colors
    y = double(y(:));
    C = zeros(numel(y),3);
    C(y==0,:) = repmat(c0, sum(y==0), 1);
    C(y~=0,:) = repmat(c1, sum(y~=0), 1);
end

function scatter_jitter(x, y, cls, c0, c1)
    % simple jitter to reduce overlap
    x = double(x(:)); y = double(y(:)); cls = double(cls(:));
    j = 0.02*randn(size(x));
    hold on; grid on;
    plot(x(cls==0)+j(cls==0), y(cls==0), '.', 'Color', c0);
    plot(x(cls==1)+j(cls==1), y(cls==1), '.', 'Color', c1);
end
end
