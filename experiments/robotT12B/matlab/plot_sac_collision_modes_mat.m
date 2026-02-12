function plot_sac_collision_modes_mat(matPath)
%PLOT_SAC_COLLISION_MODES_MAT Plot stop vs cdf curves exported by compare_sac_collision_modes.py
%
% Usage:
%   plot_sac_collision_modes_mat('D:\\VirtualSpace\\rl_mppi\\experiments\\results\\sac_collision_compare_overlay.mat')
%   plot_sac_collision_modes_mat()  % uses default relative path if exists

if nargin < 1 || isempty(matPath)
    % Default matches compare_sac_collision_modes.py output
    matPath = fullfile('..', 'results', 'sac_collision_compare_overlay.mat');
end

S = load(matPath);

labelA = string(S.label_a);
labelB = string(S.label_b);

% Episode return
fig = figure('Color', 'w', 'Position', [100 100 1100 750]);

tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile;
plot(S.a_episode_end_step, S.a_episode_return, 'LineWidth', 1.2); hold on;
plot(S.b_episode_end_step, S.b_episode_return, 'LineWidth', 1.2);
grid on; xlabel('env step'); ylabel('return'); title('Episode return');
legend(labelA, labelB, 'Location', 'best');

% Eval mean dist (±1 std)
nexttile;
plot(S.a_eval_step, S.a_eval_mean_dist, 'LineWidth', 1.6); hold on;
fill_between_std(S.a_eval_step, S.a_eval_mean_dist, S.a_eval_std_dist, 0.15);

plot(S.b_eval_step, S.b_eval_mean_dist, 'LineWidth', 1.6);
fill_between_std(S.b_eval_step, S.b_eval_mean_dist, S.b_eval_std_dist, 0.15);

grid on; xlabel('env step'); ylabel('dist'); title('Eval mean dist (\pm1 std)');
legend(labelA + " mean", labelB + " mean", 'Location', 'best');

% Rates
nexttile;
plot(S.a_eval_step, 100*S.a_eval_success_rate, 'LineWidth', 1.6); hold on;
plot(S.a_eval_step, 100*S.a_eval_success_no_collision_rate, 'LineWidth', 1.6);
plot(S.a_eval_step, 100*S.a_eval_collision_rate, 'LineWidth', 1.6);

plot(S.b_eval_step, 100*S.b_eval_success_rate, 'LineWidth', 1.6);
plot(S.b_eval_step, 100*S.b_eval_success_no_collision_rate, 'LineWidth', 1.6);
plot(S.b_eval_step, 100*S.b_eval_collision_rate, 'LineWidth', 1.6);

grid on; xlabel('env step'); ylabel('rate (%)'); title('Eval rates');
legend([
    labelA+" success", labelA+" success\_no\_coll", labelA+" collision", ...
    labelB+" success", labelB+" success\_no\_coll", labelB+" collision"], ...
    'Location', 'best', 'NumColumns', 2);

% Alpha
nexttile;
plot(S.a_eval_step, S.a_eval_alpha, 'LineWidth', 1.6); hold on;
plot(S.b_eval_step, S.b_eval_alpha, 'LineWidth', 1.6);
grid on; xlabel('env step'); ylabel('alpha'); title('Alpha');
legend(labelA, labelB, 'Location', 'best');

sgtitle("SAC collision modes: " + labelA + " vs " + labelB);

end

function fill_between_std(x, y, s, faceAlpha)
% Helper to draw y ± s as a transparent patch.
if isempty(x) || isempty(y) || isempty(s)
    return;
end
x = x(:); y = y(:); s = s(:);
if numel(x) ~= numel(y) || numel(x) ~= numel(s)
    return;
end
if ~any(s > 0)
    return;
end

xu = [x; flipud(x)];
yu = [y - s; flipud(y + s)];
ph = patch(xu, yu, 'k', 'EdgeColor', 'none');
ph.FaceAlpha = faceAlpha;
ph.FaceColor = [0 0 0];
% The patch uses black; it will blend with line colors in typical MATLAB defaults.
end
