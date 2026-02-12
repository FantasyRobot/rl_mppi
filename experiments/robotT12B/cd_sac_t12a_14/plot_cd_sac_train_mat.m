function plot_cd_sac_train_mat(matPath)
%PLOT_CD_SAC_T12A_14_TRAIN_MAT Plot CD-SAC (TD-CD) training curves from a .mat log.
%
% The .mat log is exported by train_cd_sac_t12a_14_online.py when SciPy is available.
%
% Usage:
%   plot_cd_sac_t12a_14_train_mat('D:\\VirtualSpace\\rl_mppi\\experiments\\results\\cd_sac_t12a_14_model_online_train_log.mat')
%   plot_cd_sac_t12a_14_train_mat()  % default relative path if exists

if nargin < 1 || isempty(matPath)
    matPath = fullfile('..', '..', 'results', 'cd_sac_t12a_14_model_online_train_log.mat');
end

S = load(matPath);

FS = 18;
LW1 = 2.4;   % was 1.2
LW2 = 3.2;   % was 1.6

fig = figure('Color','w','Position',[100 100 1100 820]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% Episode return
nexttile;
hRet = plot(S.episode_end_step, S.episode_return, 'LineWidth', LW1);
grid on; xlabel('env step'); ylabel('return');
title('(a) episode return');
set(gca, 'FontSize', FS);

% Eval distance (±1 std) + avg_steps on right axis
nexttile;
hDist = plot(S.eval_step, S.eval_avg_final_dist, 'LineWidth', LW2); hold on;
fill_between_std(S.eval_step, S.eval_avg_final_dist, S.eval_std_final_dist, 0.2);

ylabel('dist'); xlabel('env step'); grid on;
title('(b) eval distance');

hasSteps = isfield(S,'eval_avg_steps') && numel(S.eval_avg_steps)==numel(S.eval_step);
if hasSteps
    yyaxis right;
    hSteps = plot(S.eval_step, S.eval_avg_steps, 'LineWidth', LW1);
    ylabel('steps');
    yyaxis left;
end

if hasSteps
    legend([hDist, hSteps], {'avg\_final\_dist', 'avg\_steps'}, 'Location', 'best');
else
    legend(hDist, {'avg\_final\_dist'}, 'Location', 'best');
end
set(gca, 'FontSize', FS);

% Rates
nexttile;
plot(S.eval_step, 100*S.eval_success_rate, 'LineWidth', LW2); hold on;
plot(S.eval_step, 100*S.eval_violation_rate, 'LineWidth', LW2);
grid on; xlabel('env step'); ylabel('rate (%)');
title('(c) eval rates');
legend('success\_rate', 'violation\_rate', 'Location', 'best');
set(gca, 'FontSize', FS);

% Eval reward / alpha (twin axis)
nexttile;
hReward = plot(S.eval_step, S.eval_avg_reward, 'LineWidth', LW2); grid on; hold on;
xlabel('env step'); ylabel('avg\_reward');
title('(d) eval reward / alpha');

yyaxis right;
hAlpha = plot(S.eval_step, S.eval_alpha, 'LineWidth', LW1);
ylabel('alpha');

yyaxis left;
legend([hReward, hAlpha], {'avg\_reward', 'alpha'}, 'Location', 'best');
set(gca, 'FontSize', FS);

%sgtitle('CD-SAC T12A14 training curves');
%set(findall(fig, '-property', 'FontSize'), 'FontSize', FS);

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
end
