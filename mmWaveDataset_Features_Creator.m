%% mmWaveDataset_Features_Creator.m
% Build one dataset from unified sources (CORE + CATALOG) with DWPT images
% and 15-D Haar DWT features, and save flat arrays for Python.
%
% Key fixes:
%   • Default BALANCE_MODE = 'none'  → keeps ALL samples (~10,554)
%   • Optional 'gentle' alpha-cap or 'strict' per-bin parity
%   • Saves to a NEW file name by default to avoid loading the old 9k
%   • Writes flat arrays at root: cnnInput, cyclo_f, labels, fc, snr, src
%
% Python loader: mmwave_loader_unified.py

clc; clear; close all; rng(22);

%% ---------------- PATHS / FILES ----------------
coreFile   = 'mmWave_TxRx_Core_SC_FDMA.mat';     % Part A
baseDir    = 'C:\Users\NOMFUNDO\Documents\MATLAB\Project2\DeepSense6G';
idxCSV     = fullfile(baseDir,'_data','index.csv');    % Part B
traceDir   = fullfile(baseDir,'_data','traces');

% Save as FULL to avoid confusion with older balanced files
saveName   = 'mmWaveDataset_Balanced_DWPT_fromUnified_FULL.mat';

%% ---------------- GLOBAL PARAMS ----------------
% Wavelet / TF image
waveletName   = 'haar';
waveletLevels = 4;
TF_OUT        = [64,64];

% Windowing for Part B (match CORE rx_noCP length 1024)
WIN_SAMPLES   = 1024;
HOP_SAMPLES   = 1024;     % non-overlapping

% Balancing mode:
%   'none'   -> keep all samples (default, recommended)
%   'gentle' -> cap majority per (fc,SNR,source) bin to alpha× minority
%   'strict' -> equalize counts per bin (min(P,A))
BALANCE_MODE  = 'none';    % 'none' | 'gentle' | 'strict'
ALPHA_CAP     = 1.5;       % used only if BALANCE_MODE='gentle'
MAX_PER_BIN   = [];        % optional hard cap (applies to gentle/strict)

% Interpolation aug (off by default)
DO_INTERPOLATION = false;

fprintf('=== BUILD FROM UNIFIED SOURCES ===\n');

%% =======================================================================
%% LOAD & CONVERT CORE (PART A)
%% =======================================================================
assert(isfile(coreFile), 'Core file not found: %s', coreFile);
S = load(coreFile);
assert(isfield(S,'core'),'Core file missing variable "core".');
core = S.core;
fprintf('Loaded core: %d entries\n', numel(core));

dataset = struct([]); did = 1;

for k = 1:numel(core)
    if ~isfield(core(k),'rx_noCP') || isempty(core(k).rx_noCP), continue; end
    rx = core(k).rx_noCP(:);
    if any(~isfinite(rx)), continue; end

    x_mag = abs(rx);  % magnitude for features/images

    % 15-D features + DWPT image
    [cyclo_vec, approx, details] = compute_cyclo_features_dwt_only( ...
        x_mag, waveletLevels, waveletName, core(k).class);
    cnnImg = dwpt_tf_image_fast(x_mag, waveletName, waveletLevels, TF_OUT);

    dataset(did).source    = 'CORE';
    dataset(did).fc        = double(getfield_safe(core(k),'fc_Hz',NaN));
    dataset(did).SNRdB     = double(getfield_safe(core(k),'SNR_dB',NaN));
    dataset(did).class     = int16(core(k).class); % 1/0
    dataset(did).label     = string(getfield_safe(core(k),'label','CORE'));
    dataset(did).rx        = rx;
    dataset(did).cyclo     = single(cyclo_vec);
    dataset(did).cnnInput  = single(cnnImg);
    dataset(did).wavelet.approx  = approx;
    dataset(did).wavelet.details = details;
    did = did + 1;
end

fprintf('Core → dataset entries: %d\n', did-1);

%% =======================================================================
%% INGEST CATALOG (PART B)
%% =======================================================================
if isfile(idxCSV)
    T = readtable(idxCSV);
    need = {'file','fs','fcHz','source','tag'};
    ok = all(ismember(need, T.Properties.VariableNames));
    assert(ok, 'index.csv must have columns: %s', strjoin(need,', '));

    nTr = height(T);
    fprintf('Catalog traces: %d\n', nTr);

    for i = 1:nTr
        % Robust field reads
        fp  = as_char(T.file(i));
        fs  = as_double(T.fs(i)); %#ok<NASGU>
        fc  = as_double(T.fcHz(i));
        src = string(T.source(i));
        tag = string(T.tag(i));

        % Normalize/repair path if relative
        if ~isfile(fp)
            tail = char(extractAfter(string(fp), 'traces/'));
            fp2  = fullfile(traceDir, tail);
            if isfile(fp2), fp = fp2; else, warning('Missing trace: %s', fp); continue; end
        end

        Sx = load(fp);

        % IQ preferred; else analytic IQ from power x/xsim
        iq = [];
        if isfield(Sx,'iq'), iq = Sx.iq(:); end
        if isempty(iq)
            xpow = [];
            if isfield(Sx,'x'),    xpow = Sx.x(:);    end  % DeepSense real
            if isempty(xpow) && isfield(Sx,'xsim'), xpow = Sx.xsim(:); end % Sim snapshot
            if isempty(xpow)
                warning('No iq/x/xsim in %s; skipping.', fp); continue;
            end
            iq = make_analytic_from_power(xpow);
        end
        iq = complex(iq(:));
        if numel(iq) < WIN_SAMPLES || any(~isfinite(iq)), continue; end

        % Window into clips & create matched ABSENT
        starts = 1:HOP_SAMPLES:(numel(iq)-WIN_SAMPLES+1);
        for s = 1:numel(starts)
            seg = iq(starts(s):starts(s)+WIN_SAMPLES-1);
            if any(~isfinite(seg)), continue; end

            % PRESENT
            x_mag = abs(seg);
            [cvP, apP, dtP] = compute_cyclo_features_dwt_only(x_mag, waveletLevels, waveletName, 1);
            imgP = dwpt_tf_image_fast(x_mag, waveletName, waveletLevels, TF_OUT);

            dataset(did).source   = char(src);
            dataset(did).fc       = fc;
            dataset(did).SNRdB = 1e9;   % unknown SNR sentinel
            dataset(did).class    = int16(1);
            dataset(did).label    = char(tag);
            dataset(did).rx       = seg;
            dataset(did).cyclo    = single(cvP);
            dataset(did).cnnInput = single(imgP);
            dataset(did).wavelet.approx  = apP;
            dataset(did).wavelet.details = dtP;
            did = did + 1;

            % ABSENT: matched-power AWGN
            pwr = mean(abs(seg).^2);
            n   = sqrt(pwr/2) * (randn(size(seg)) + 1j*randn(size(seg)));
            xn  = abs(n);
            [cvA, apA, dtA] = compute_cyclo_features_dwt_only(xn, waveletLevels, waveletName, 0);
            imgA = dwpt_tf_image_fast(xn, waveletName, waveletLevels, TF_OUT);

            dataset(did).source   = [char(src) '_ABS'];
            dataset(did).fc       = fc;
            dataset(did).SNRdB = 1e9;   % unknown SNR sentinel
            dataset(did).class    = int16(0);
            dataset(did).label    = 'ABSENT(gen)';
            dataset(did).rx       = n;
            dataset(did).cyclo    = single(cvA);
            dataset(did).cnnInput = single(imgA);
            dataset(did).wavelet.approx  = apA;
            dataset(did).wavelet.details = dtA;
            did = did + 1;
        end
    end
else
    warning('Index CSV not found: %s  (skipping Part B)', idxCSV);
end

fprintf('Merged dataset size (core + catalog): %d\n', numel(dataset));

%% =======================================================================
%% (OPTIONAL) INTERPOLATION AUGMENTATION
%% =======================================================================
if DO_INTERPOLATION
    % (left as in your previous version)
end

%% =======================================================================
%% BALANCING (none / gentle / strict)
%% =======================================================================
switch lower(string(BALANCE_MODE))
    case "none"
        fprintf('Balancing: NONE (keeping all samples)\n');

    case "gentle"
        fprintf('Balancing: GENTLE alpha-cap (alpha=%.2f) per (fc,SNR,source)\n', ALPHA_CAP);
        dataset = gentle_balance(dataset, ALPHA_CAP, MAX_PER_BIN);
        fprintf('Gently balanced dataset size: %d\n', numel(dataset));

    case "strict"
        fprintf('Balancing: STRICT per (fc,SNR,source)\n');
        dataset = strict_balance(dataset, MAX_PER_BIN);
        fprintf('Strictly balanced dataset size: %d\n', numel(dataset));

    otherwise
        error('Unknown BALANCE_MODE: %s', BALANCE_MODE);
end

%% =======================================================================
%% PACK & SAVE (flat arrays for Python/MATLAB)
%% =======================================================================
N = numel(dataset);
labels = int16([dataset.class].'); labels(labels~=0)=1;     % Nx1
fc      = double([dataset.fc].');                           % Nx1
snr     = double([dataset.SNRdB].');                        % Nx1
src     = string({dataset.source}).';                       % Nx1

cyclo_f = zeros(N,15,'single');
for i = 1:N
    v = single(dataset(i).cyclo);
    cyclo_f(i,:) = v(1:15);   % keep first 15 features (ModIndex is last)
end

H = TF_OUT(1); W = TF_OUT(2);
cnnInput = zeros(N,1,H,W,'single');
for i = 1:N
    cnnInput(i,1,:,:) = single(dataset(i).cnnInput);
end

save(saveName,'dataset','labels','cyclo_f','cnnInput','fc','snr','src','-v7.3');
fprintf('✅ Saved merged dataset: %s\n', saveName);

%% =======================================================================
%% HELPERS
%% =======================================================================
function v = getfield_safe(s, f, def)
    if isstruct(s) && isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = def; end
end

function s = as_char(x)
    if iscell(x), s = char(string(x{1}));
    else,         s = char(string(x));
    end
end

function d = as_double(x)
    if iscell(x), d = double(x{1});
    else,         d = double(x);
    end
end

function [cyclo_vector, approx, details] = compute_cyclo_features_dwt_only(x_mag, L, wname, classFlag)
    if nargin<4, classFlag=1; end
    x_mag = double(x_mag(:));
    [c,l] = wavedec(x_mag, L, wname);
    details = cell(1,L);
    for k = 1:L
        d = detcoef(c,l,k); if isempty(d), d = 0; end
        details{k} = d(:);
    end
    approx = appcoef(c,l,wname,L);

    % 1) Normalized energies (4 bands)
    eBands = cellfun(@(d) sum(abs(d).^2,'omitnan'), details);
    eTot   = sum(eBands) + eps;
    eNorm  = eBands / eTot; eNorm(~isfinite(eNorm)) = 0;

    % 2) TKEO mean per band
    TKEOmean = zeros(1,L);
    for k = 1:L
        d = details{k}; if numel(d) < 3, TKEOmean(k)=0; continue; end
        psi = d(2:end-1).^2 - d(1:end-2).*d(3:end);
        TKEOmean(k) = mean(abs(psi),'omitnan');
    end

    % 3) Gini per band
    Gini = zeros(1,L);
    for k = 1:L
        v = abs(details{k}); n = numel(v);
        if n==0, Gini(k)=0; continue; end
        v = sort(v(:)); denom = sum(v)+eps;
        Gini(k) = 1 - 2 * sum(((n+1-(1:n)).'/n).*v) / denom;
        if ~isfinite(Gini(k)), Gini(k)=0; end
    end

    % 4) ModDepth at scale 1
    ModDepth1 = 0;
    if L>=1
        a = abs(details{1}); if numel(a)>=2
            m = 2*floor(numel(a)/2); a = a(1:m);
            hd = (a(1:2:end) - a(2:2:end)) / sqrt(2);
            ModDepth1 = sum(hd.^2) / (sum(a.^2) + eps);
        end
    end

    % 5) Entropy over normalized energies
    Entropy = -sum(eNorm .* log2(eNorm + eps));

    % 6) ModIndex (gated)
    if classFlag==0, modIndex=0; else, [~,modIndex]=max(eNorm); end

    cyclo_vector = single([eNorm, TKEOmean, Gini, ModDepth1, Entropy, modIndex]);
    cyclo_vector(~isfinite(cyclo_vector)) = 0;
end

function I = dwpt_tf_image_fast(x, wname, L, outSz)
    x = double(x(:));
    T = wpdec(x, L, wname);
    nodes = leaves(T);
    C = cell(numel(nodes),1);
    for k = 1:numel(nodes), C{k} = wpcoef(T, nodes(k)); end
    lens = cellfun(@numel, C); Lmin = min(lens);
    for k = 1:numel(C), if numel(C{k})>Lmin, C{k}=C{k}(1:Lmin); end, end
    E = cell2mat(cellfun(@(v) v(:).', C, 'UniformOutput', false));
    P = E.^2; E = log1p(bsxfun(@rdivide, P, max(P,[],2)+eps));
    E = E - min(E(:)); mx = max(E(:)); if mx>0, E = E./mx; end
    I = imresize(E, outSz);
end

function iq = make_analytic_from_power(p)
    p = double(p(:));
    a = sqrt(max(p,0)); a(a==0) = 1e-12;
    try
        iq = hilbert(a);
    catch
        % small random phase to avoid degeneracy if hilbert unavailable
        phi = unwrap(angle(hilbert(a + 1e-6*randn(size(a)))));
        iq  = a .* exp(1j*phi);
    end
    % normalize avg power
    iq = iq / sqrt(mean(abs(iq).^2)+eps);
end

%% ---------- Balancing helpers ----------
function dataset = strict_balance(dataset, MAX_PER_BIN)
    fcAll  = double([dataset.fc]);
    snrAll = double([dataset.SNRdB]); snrAll(isnan(snrAll)) = 1e9; % bucket unknown SNRs
    srcAll = string({dataset.source}).';
    clsAll = double([dataset.class]);

    fcStr  = compose('%.6e', fcAll(:));
    snrStr = compose('%.3f',  snrAll(:));
    allKeys = strcat(fcStr, '|', snrStr, '|', srcAll);

    uKeys = unique(allKeys);
    keep = false(1, numel(dataset));
    for kk = 1:numel(uKeys)
        idx = find(allKeys==uKeys(kk));
        if numel(idx) < 2, continue; end
        iP = idx(clsAll(idx)==1); iA = idx(clsAll(idx)==0);
        if isempty(iP) || isempty(iA), continue; end
        nTake = min(numel(iP), numel(iA));
        if ~isempty(MAX_PER_BIN), nTake = min(nTake, MAX_PER_BIN); end
        keep(iP(randperm(numel(iP), nTake))) = true;
        keep(iA(randperm(numel(iA), nTake))) = true;
    end
    if any(keep), dataset = dataset(keep); end
end

function dataset = gentle_balance(dataset, alpha, MAX_PER_BIN)
    fcAll  = double([dataset.fc]);
    snrAll = double([dataset.SNRdB]); snrAll(isnan(snrAll)) = 1e9; % bucket unknown SNRs
    srcAll = string({dataset.source}).';
    clsAll = double([dataset.class]);

    fcStr  = compose('%.6e', fcAll(:));
    snrStr = compose('%.3f',  snrAll(:));
    allKeys = strcat(fcStr, '|', snrStr, '|', srcAll);

    uKeys = unique(allKeys);
    keep = false(1, numel(dataset));
    for kk = 1:numel(uKeys)
        idx = find(allKeys==uKeys(kk));
        if numel(idx) < 2, continue; end
        iP = idx(clsAll(idx)==1); iA = idx(clsAll(idx)==0);
        if isempty(iP) || isempty(iA), continue; end

        nMin   = min(numel(iP), numel(iA));
        nMajCap= ceil(alpha * nMin);
        if numel(iP) >= numel(iA)
            takeP = min(numel(iP), nMajCap); takeA = numel(iA);
        else
            takeA = min(numel(iA), nMajCap); takeP = numel(iP);
        end
        if ~isempty(MAX_PER_BIN)
            takeP = min(takeP, MAX_PER_BIN);
            takeA = min(takeA, MAX_PER_BIN);
        end
        keep(iP(randperm(numel(iP), takeP))) = true;
        keep(iA(randperm(numel(iA), takeA))) = true;
    end
    if any(keep), dataset = dataset(keep); end
end
