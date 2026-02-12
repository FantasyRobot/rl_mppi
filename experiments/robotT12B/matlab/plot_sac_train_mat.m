function plot_sac_train_mat(mat_path, out_png)
% Plot SAC / CDF-SAC training curves from exported .mat.
%
% Usage examples:
%   plot_sac_train_mat('models/sac_t12a_14_model_train_log.mat');
%   plot_sac_train_mat('results/sac_collision_compare_cdf_train_log.mat', 'cdf_sac_train.png');

if nargin < 1 || isempty(mat_path)
    error('mat_path is required');
end

S = load(mat_path);

if nargin < 2 || isempty(out_png)
    [p, n, ~] = fileparts(mat_path);
    out_png = fullfile(p, [n '_plots.png']);
end

FS = 18;
LW = 2.4;

fig = figure('Color','w','Position',[100 100 1200 800]);

% ---- (1) Episode return ----
subplot(2,2,1);
if isfield(S,'episode_end_step') && isfield(S,'episode_return') && ~isempty(S.episode_end_step)
    plot(double(S.episode_end_step), double(S.episode_return), 'LineWidth', LW);
end
xlabel('env step'); ylabel('return');
title('(a) Episode Return');
grid on;
set(gca,'FontSize',FS);

% ---- (2) Eval mean dist (+/- std) + mean steps (right axis) ----
subplot(2,2,2);
h = [];
leg = {};

x = []; y = []; ys = [];
if isfield(S,'eval_step'), x = double(S.eval_step(:)); end
if isfield(S,'eval_mean_dist'), y = double(S.eval_mean_dist(:)); end
if isfield(S,'eval_std_dist'), ys = double(S.eval_std_dist(:)); end

if ~isempty(x) && ~isempty(y)
    yyaxis left;
    hold on;
    if ~isempty(ys) && numel(ys) == numel(y) && any(ys > 0)
        xx = [x; flipud(x)];
        yy = [y - ys; flipud(y + ys)];
        fill(xx, yy, [0 0.4470 0.7410], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    end
    h1 = plot(x, y, 'LineWidth', LW);
    h = [h h1]; leg = [leg {'mean dist'}];
    ylabel('dist');
end

if isfield(S,'eval_mean_steps') && ~isempty(x) && numel(S.eval_mean_steps) == numel(x)
    yyaxis right;
    h2 = plot(x, double(S.eval_mean_steps(:)), 'LineWidth', LW, 'Color', [0.8500 0.3250 0.0980]);
    h = [h h2]; leg = [leg {'mean steps'}];
    ylabel('steps');
end

xlabel('env step');
title('(b) Eval Mean Dist');
grid on;
set(gca,'FontSize',FS);
if ~isempty(h)
    legend(h, leg, 'Location','best');
end

% ---- (3) Eval rates ----
subplot(2,2,3);
h = []; leg = {};
if isfield(S,'eval_step'), x = double(S.eval_step(:)); else, x = []; end

if ~isempty(x) && isfield(S,'eval_success_rate') && numel(S.eval_success_rate) == numel(x)
    h1 = plot(x, 100*double(S.eval_success_rate(:)), 'LineWidth', LW); hold on;
    h = [h h1]; leg = [leg {'success'}];
end
if ~isempty(x) && isfield(S,'eval_success_no_collision_rate') && numel(S.eval_success_no_collision_rate) == numel(x)
    h2 = plot(x, 100*double(S.eval_success_no_collision_rate(:)), 'LineWidth', LW);
    h = [h h2]; leg = [leg {'success no-collision'}];
end
if ~isempty(x) && isfield(S,'eval_collision_rate') && numel(S.eval_collision_rate) == numel(x)
    h3 = plot(x, 100*double(S.eval_collision_rate(:)), 'LineWidth', LW);
    h = [h h3]; leg = [leg {'collision'}];
end

xlabel('env step'); ylabel('rate (%)');
title('(c) Eval Success/Collision Rates');
grid on;
set(gca,'FontSize',FS);
if ~isempty(h)
    legend(h, leg, 'Location','best');
end

% ---- (4) Alpha ----
subplot(2,2,4);
if isfield(S,'eval_step') && isfield(S,'eval_alpha') && ~isempty(S.eval_step)
    plot(double(S.eval_step(:)), double(S.eval_alpha(:)), 'LineWidth', LW);
end
xlabel('env step'); ylabel('alpha');
title('(d) Alpha');
grid on;
set(gca,'FontSize',FS);

% Global title
sgtitle_str = 'SAC training curves';
if isfield(S,'collision_mode')
    try
        %sgtitle_str = sprintf('SAC training curves (collision_mode=%s)', string(S.collision_mode));
    catch
    end
end
try
    %sgtitle(sgtitle_str, 'FontSize', FS);
catch
end

exportgraphics(fig, out_png, 'Resolution', 200);
fprintf('[PLOT] saved: %s\n', out_png);

end
