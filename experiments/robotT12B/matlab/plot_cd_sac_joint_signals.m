function plot_cd_sac_t12a_14_joint_signals(basePath)
% Plot per-joint qvel/qacc exported by cd_sac_t12a_14_cli.py test --viewer.
%
% Usage:
%   plot_cd_sac_t12a_14_joint_signals('..\results\cd_sac_t12a_14_joint')
% This will load:
%   <basePath>_qvel.csv
%   <basePath>_qacc.csv
%   <basePath>_meta.txt

if nargin < 1
    error('basePath required, e.g. ..\\results\\cd_sac_t12a_14_joint');
end

qvelPath = basePath + "_qvel.csv";
qaccPath = basePath + "_qacc.csv";
metaPath = basePath + "_meta.txt";

qvel = readmatrix(qvelPath);
qacc = readmatrix(qaccPath);

numJoints = min(6, size(qvel, 2));
leg = strings(1, numJoints);
for i = 1:numJoints
    leg(i) = "joint " + string(i);
end

dt = 1.0;
if isfile(metaPath)
    txt = fileread(metaPath);
    tok = regexp(txt, 'dt_eff:\s*([0-9eE\+\-\.]+)', 'tokens', 'once');
    if ~isempty(tok)
        dt = str2double(tok{1});
    end
end

tVel = (0:size(qvel,1)-1) * dt;
tAcc = (0:size(qacc,1)-1) * dt;

subplot(2,1,1);
plot(tVel, qvel, 'LineWidth', 1.0);
plot(tVel, qvel(:, 1:numJoints), 'LineWidth', 1.0);
xlabel('time (s)');
xlabel('time (s)', 'FontSize', 14);
ylabel('qvel (rad/s)', 'FontSize', 14);
title('(a) Joint Velocities', 'FontSize', 14);
set(gca, 'FontSize', 14);
legend(leg, 'FontSize', 14, 'Location', 'best');
grid on;

subplot(2,1,2);
plot(tAcc, qacc, 'LineWidth', 1.0);
plot(tAcc, qacc(:, 1:numJoints), 'LineWidth', 1.0);
xlabel('time (s)');
xlabel('time (s)', 'FontSize', 14);
ylabel('qacc (rad^2/s)', 'FontSize', 14);
title('(b) Joint Accelerations', 'FontSize', 14);
set(gca, 'FontSize', 14);
legend(leg, 'FontSize', 14, 'Location', 'best');
grid on;
end
