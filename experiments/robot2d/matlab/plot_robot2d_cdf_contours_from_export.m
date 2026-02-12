function plot_robot2d_cdf_contours_from_export(path, outPng)
%PLOT_ROBOT2D_CDF_CONTOURS_FROM_EXPORT Plot Robot2D CDF contours from exported .mat or .h5.
%
% Exported by:
%   python experiments/robot2d/test_robot2d_cdf_contours_and_collision.py --export_mat robot2d_cdf_export.mat
%   python experiments/robot2d/test_robot2d_cdf_contours_and_collision.py --export_mat robot2d_cdf_export.h5
%
% Usage:
%   addpath('D:/VirtualSpace/rl_mppi/experiments/robot2d/matlab')
%   plot_robot2d_cdf_contours_from_export('D:/VirtualSpace/rl_mppi/experiments/results/robot2d_cdf_export.mat')
%   plot_robot2d_cdf_contours_from_export('.../robot2d_cdf_export.h5', 'robot2d_matlab.png')

if nargin < 2
    outPng = '';
end

path = char(path);
[~,~,ext] = fileparts(path);
ext = lower(ext);

if strcmp(ext,'.mat')
    S = load(path);
elseif strcmp(ext,'.h5') || strcmp(ext,'.hdf5')
    S = load_h5_struct(path);
else
    error('Unsupported extension: %s (expected .mat or .h5)', ext);
end

req = {'q0','q1','cdf','q_spline_line','q_contour_line','eef_spline','eef_contour','obstacles','target_pos','start_level'};
for i = 1:numel(req)
    if ~isfield(S, req{i})
        error('Missing field in export: %s', req{i});
    end
end

q0 = S.q0;
q1 = S.q1;
d  = S.cdf;
qSplineLine = S.q_spline_line;
qContourLine = S.q_contour_line;
eefSpline = S.eef_spline;
eefContour = S.eef_contour;
obs = S.obstacles;
target = S.target_pos;
startLevel = S.start_level;

% Optional fields (newer exports may include these)
if isfield(S, 'q_spline')
    qSpline = S.q_spline;
else
    qSpline = [];
end
if isfield(S, 'q_contour')
    qContour = S.q_contour;
else
    qContour = [];
end
if isfield(S, 'link_lengths')
    linkLengths = double(S.link_lengths(:)');
else
    linkLengths = [2.0, 2.0];
end
if numel(linkLengths) < 2
    linkLengths = [2.0, 2.0];
end
l1 = linkLengths(1);
l2 = linkLengths(2);

% Style (keep consistent across subplots)
cSpline = [0.1216 0.4667 0.7059];  % blue
cCDF    = [0.2 0.7 0.2];           % green
cStart  = [1.0 0.0 0.0];           % red
cGoal   = [1.0 0.0 0.0];           % red
cColl   = [1.0 0.0 0.0];
fontSize = 18;

% Keep trajectory line widths consistent across subplots.
lwSpline = 2.2;
lwCDF    = 2.2;
lwColl   = 2.2;

fig = figure('Color','w','Position',[100,100,1400,650]);

% --- Left: C-space contours
ax1 = subplot(1,2,1);
hold(ax1,'on');
axis(ax1,'equal');
grid(ax1,'on');
set(ax1,'XLim',[-pi, pi],'YLim',[-pi, pi]);
title(ax1,'(a) CDF空间');
xlabel(ax1,'q_1'); ylabel(ax1,'q_2');

levels = 18;
contourf(ax1, q0, q1, d, levels, 'LineStyle','none');
% Use a colored colormap (avoid grayscale which can look flat).
colormap(ax1, coolwarm_colormap(256));
cb = colorbar(ax1);
cb.Label.String = 'cdf(q)';

contour(ax1, q0, q1, d, [0,0], 'LineColor','k', 'LineWidth',2.0);


hSplineC = plot(ax1, qSplineLine(:,1), qSplineLine(:,2), '-', 'Color', cSpline, 'LineWidth',lwSpline);
hStartC  = plot(ax1, qSplineLine(1,1), qSplineLine(1,2), 's', 'Color', cStart, 'MarkerFaceColor', cStart, 'MarkerSize',7);
hLevelC  = plot(ax1, qContourLine(:,1), qContourLine(:,2), '-', 'Color', cCDF, 'LineWidth',lwCDF);

% Goal (IK) markers in configuration space (2-link analytic IK)
hGoalIK = [];
ikSols = ik_2link_solutions(double(target(:))', l1, l2);
ikSols = ikSols(1,:);
if ~isempty(ikSols)
    for k = 1:size(ikSols,1)
        if k == 1
            hGoalIK = plot(ax1, ikSols(k,1), ikSols(k,2), 'x', 'Color', cGoal, 'MarkerSize',10, 'LineWidth',2);
        else
            plot(ax1, ikSols(k,1), ikSols(k,2), 'mx', 'MarkerSize',10, 'LineWidth',2, 'HandleVisibility','off');
        end
    end
end

% Collision markers (C-space): match Python (spline only)
hColC = [];
if isfield(S, 'collided_spline') && ~isempty(qSpline)
    hit = find(double(S.collided_spline(:)) ~= 0);
    if ~isempty(hit)
        pts = qSpline(hit, :);
        hColC = plot(ax1, pts(:,1), pts(:,2), '-', 'Color', cColl, 'LineWidth',lwColl, 'MarkerSize',6);
    end
end

% CDF trajectory point cloud (C-space)
hCDFPts = [];
if ~isempty(qContour)
    % Use the same color as the CDF trajectory (avoid blending to near-white).
    hCDFPts = plot(ax1, qContour(:,1), qContour(:,2), '-', 'Color', cCDF, 'LineWidth',lwCDF, 'MarkerSize',10);
end

% Explicit legend to avoid MATLAB auto dataN entries and match Python ordering/text.
handlesC = [ hStartC];
labelsC = {'起始点'};
if ~isempty(hGoalIK)
    handlesC(end+1) = hGoalIK; %#ok<AGROW>
    labelsC{end+1} = '目标点'; %#ok<AGROW>
end
handlesC(end+1) = hSplineC;
labelsC{end+1} = '样条轨迹';
if ~isempty(hColC)
    handlesC(end+1) = hColC; %#ok<AGROW>
    labelsC{end+1} = '碰撞轨迹'; %#ok<AGROW>
end

%handlesC(end+1) = hLevelC;
%labelsC{end+1} = 'Level match -> CW slide';

if ~isempty(hCDFPts)
    handlesC(end+1) = hCDFPts; %#ok<AGROW>
    labelsC{end+1} = 'CDF轨迹'; %#ok<AGROW>
end


legend(ax1, handlesC, labelsC, 'Location','southwest');

% --- Right: task-space
ax2 = subplot(1,2,2);
hold(ax2,'on');
axis(ax2,'equal');
grid(ax2,'on');
title(ax2,'(b) 工作空间');
xlabel(ax2,'x'); ylabel(ax2,'y');

if ~isempty(obs)
    for i = 1:size(obs,1)
        x = obs(i,1); y = obs(i,2); r = obs(i,3);
        draw_circle_patch(ax2, x, y, r, [0.6 0.6 0.6], 0.35);
    end
end

% Plot trajectories first so start/goal markers can be drawn on top.
hSplineT = plot(ax2, eefSpline(:,1), eefSpline(:,2), '-', 'Color', cSpline, 'LineWidth',lwSpline, 'DisplayName','样条轨迹');
hCDFT    = plot(ax2, eefContour(:,1), eefContour(:,2), '-', 'Color', cCDF, 'LineWidth',lwCDF, 'DisplayName','CDF轨迹');

% Collision markers (task-space): match Python (contour only)
hColT = [];
if isfield(S, 'collided_contour') && ~isempty(eefContour)
    hit2 = find(double(S.collided_contour(:)) ~= 0);
    if ~isempty(hit2)
        hColT = plot(ax2, eefContour(hit2,1), eefContour(hit2,2), '.', 'Color', cColl, 'LineWidth',lwColl, 'MarkerSize',6, 'DisplayName','碰撞点');
    end
end

% Draw start/goal last so they are on top and not occluded.
hStartT = plot(ax2, eefSpline(1,1), eefSpline(1,2), 's', 'Color', cStart, 'MarkerFaceColor', cStart, 'MarkerSize',7, 'DisplayName','起始点');
hGoalT  = plot(ax2, target(1), target(2), 'x', 'Color', cGoal, 'MarkerSize',12, 'LineWidth',2, 'DisplayName','目标点');

% Draw 2-link snapshots (Python-like)
numFrames = 14;
if ~isempty(qSpline)
    idx = unique(round(linspace(1, size(qSpline,1), max(2,numFrames))));
    for j = 1:numel(idx)
        a = 0.08 + 0.55 * ((j-1) / max(1,(numel(idx)-1)));
        draw_2link(ax2, qSpline(idx(j),:), l1, l2, cSpline, a);
    end
end
if ~isempty(qContour)
    idx2 = unique(round(linspace(1, size(qContour,1), max(2,numFrames))));
    for j = 1:numel(idx2)
        a = 0.08 + 0.55 * ((j-1) / max(1,(numel(idx2)-1)));
        draw_2link(ax2, qContour(idx2(j),:), l1, l2, cCDF, a);
    end
end

reach = l1 + l2;
set(ax2,'XLim',[-reach - 0.6, reach + 0.6], 'YLim',[-reach - 0.6, reach + 0.6]);

% Explicit legend ordering.
handlesT = [hStartT, hGoalT, hSplineT, hCDFT];
labelsT  = {'起始点', '目标点', '样条轨迹', 'CDF轨迹'};
if ~isempty(hColT)
    handlesT(end+1) = hColT; %#ok<AGROW>
    labelsT{end+1} = '碰撞点'; %#ok<AGROW>
end
legend(ax2, handlesT, labelsT, 'Location','southwest');

% Apply global font size (titles/labels/ticks/legends/colorbar, etc.).
apply_fontsize(fig, fontSize);

if ~isempty(outPng)
    if exist('exportgraphics','file') == 2
        exportgraphics(fig, outPng, 'Resolution',150);
    else
        print(fig, outPng, '-dpng', '-r150');
    end
    fprintf('[PLOT] saved: %s\n', outPng);
end

end

function apply_fontsize(fig, fontSize)
% Apply a uniform font size to all objects in the figure that support it.
objs = findall(fig, '-property', 'FontSize');
if isempty(objs)
    return;
end
try
    set(objs, 'FontSize', fontSize);
catch
    % Fallback: set individually for maximum compatibility.
    for i = 1:numel(objs)
        try
            set(objs(i), 'FontSize', fontSize);
        catch
        end
    end
end
end

function S = load_h5_struct(path)
info = h5info(path);
S = struct();
for i = 1:numel(info.Datasets)
    name = info.Datasets(i).Name;
    S.(name) = h5read(path, ['/' name]);
end
end

function cmap = coolwarm_colormap(n)
if nargin < 1
    n = 256;
end
x = linspace(0,1,n)';
b = [0.230, 0.299, 0.754];
w = [1.0, 1.0, 1.0];
r = [0.706, 0.016, 0.150];
mid = floor(n/2);
cmap1 = b + (w-b) .* (x(1:mid) / x(mid));
cmap2 = w + (r-w) .* ((x(mid+1:end)-x(mid+1)) / (1-x(mid+1)));
cmap = [cmap1; cmap2];
end

function h = draw_circle_patch(ax, x, y, r, faceColor, faceAlpha)
% Draw a filled circle using patch (works in older MATLAB).
t = linspace(0, 2*pi, 64);
xp = x + r*cos(t);
yp = y + r*sin(t);
h = patch(ax, xp, yp, faceColor, 'EdgeColor','none', 'HandleVisibility','off');
try
    set(h, 'FaceAlpha', faceAlpha);
catch
    % Very old MATLAB: ignore alpha.
end

end

function draw_2link(ax, q, l1, l2, baseColor, alphaBlend)
% Draw 2-link planar arm at configuration q=[q1,q2].
% Older MATLAB lines don't support alpha; simulate by blending with white.
q1 = double(q(1));
q2 = double(q(2));
p0 = [0.0, 0.0];
p1 = [l1*cos(q1), l1*sin(q1)];
p2 = p1 + [l2*cos(q1+q2), l2*sin(q1+q2)];

a = max(0.0, min(1.0, double(alphaBlend)));
c = a * baseColor + (1.0 - a) * [1 1 1];

plot(ax, [p0(1) p1(1) p2(1)], [p0(2) p1(2) p2(2)], '-', 'Color', c, 'LineWidth', 2.0, 'HandleVisibility','off');
plot(ax, [p0(1) p1(1) p2(1)], [p0(2) p1(2) p2(2)], 'o', 'Color', c, 'MarkerSize', 2.0, 'MarkerFaceColor', c, 'HandleVisibility','off');
end

function sols = ik_2link_solutions(target, l1, l2)
% Analytic IK for planar 2-link arm.
% Returns up to two solutions [q1,q2] in [-pi,pi], or [] if unreachable.
target = double(target(:));
x = target(1);
y = target(2);
r2 = x*x + y*y;

den = 2.0*l1*l2;
if den <= 1e-9
    sols = [];
    return;
end

c2 = (r2 - l1*l1 - l2*l2) / den;
if c2 < -1.0 - 1e-6 || c2 > 1.0 + 1e-6
    sols = [];
    return;
end

c2 = max(-1.0, min(1.0, c2));
s2a = sqrt(max(0.0, 1.0 - c2*c2));

sols = zeros(2,2);
for i = 1:2
    s2 = s2a;
    if i == 2
        s2 = -s2a;
    end
    q2 = atan2(s2, c2);
    k1 = l1 + l2*c2;
    k2 = l2*s2;
    q1 = atan2(y, x) - atan2(k2, k1);
    sols(i,:) = wrap_pi([q1, q2]);
end
end

function q = wrap_pi(q)
q = mod(q + pi, 2*pi) - pi;
end

