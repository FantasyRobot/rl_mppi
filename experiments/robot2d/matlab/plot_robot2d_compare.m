function plot_robot2d_compare(matPath, numLinkFrames)
%PLOT_ROBOT2D_COMPARE  Reproduce compare plots from exported .mat.
%
% Usage:
%   plot_robot2d_compare('D:/VirtualSpace/rl_mppi/experiments/results/robot2d_obstacle_compare.mat')
%   plot_robot2d_compare('...mat', 12)  % overlay ~12 link poses per method
%   plot_robot2d_compare  % uses default next to results

    if nargin < 2 || isempty(numLinkFrames)
        numLinkFrames = 12;
    end

    if nargin < 1 || strlength(string(matPath)) == 0
        matPath = fullfile('..','results','robot2d_obstacle_compare.mat');
    end

    S = load(matPath);

    % Required fields
    obstacles = getfield_or(S, 'obstacles', zeros(0,3)); %#ok<GFLD>
    target = getfield_or(S, 'target_pos', [0 0]);

    names = {'MPPI','SAC','RLMPPI'};
    colors = containers.Map({'MPPI','SAC','RLMPPI'}, {[0.1216 0.4667 0.7059], [0.1725 0.6275 0.1725], [1.0000 0.4980 0.0549]});
    styles = containers.Map({'MPPI','SAC','RLMPPI'}, {'-','--',':'});

    link_lengths = getfield_or(S, 'link_lengths', [2 2]);
    if numel(link_lengths) < 2
        link_lengths = [2 2];
    end
    link_lengths = link_lengths(:)';
    l1 = double(link_lengths(1));
    l2 = double(link_lengths(2));

    % --------- Figure 1: EEF trajectory (3 subplots)
    figure('Name','Robot2D Compare: Trajectory','Color','w');
    tiledlayout(1,3, 'Padding','compact', 'TileSpacing','compact');

    for i = 1:numel(names)
        nm = names{i};
        eef = getfield_or(S, nm + "_eef", zeros(0,2));
        q = getfield_or(S, nm + "_q", zeros(0,2));

        nexttile; hold on; axis equal; grid on;
        title(nm);
        xlabel('x'); ylabel('y');

        % obstacles
        for k = 1:size(obstacles,1)
            viscircles(obstacles(k,1:2), obstacles(k,3), 'Color',[0.5 0.5 0.5], 'LineWidth',1);
        end

        % eef path
        if size(eef,1) >= 1
            plot(eef(:,1), eef(:,2), styles(nm), 'Color', colors(nm), 'LineWidth', 2);
            plot(eef(1,1), eef(1,2), 'ks', 'MarkerSize', 7, 'MarkerFaceColor','k');
        end

        % overlay 2-link robot along rollout
        if size(q,1) >= 1 && size(q,2) >= 2
            kFrames = max(2, min(double(numLinkFrames), size(q,1)));
            idx = unique(round(linspace(1, size(q,1), kFrames)));
            baseColor = colors(nm);
            for jj = 1:numel(idx)
                tIdx = idx(jj);
                a = 0.15 + 0.80 * (jj / max(1, (numel(idx) - 1))); % pseudo-alpha
                c = lighten_color(baseColor, 1.0 - a);
                pts = fk_2link(q(tIdx,1), q(tIdx,2), l1, l2);
                plot(pts(:,1), pts(:,2), '-', 'Color', c, 'LineWidth', 1.5);
                plot(pts(:,1), pts(:,2), 'o', 'Color', c, 'MarkerSize', 3, 'MarkerFaceColor', c);
            end
        end
        plot(target(1), target(2), 'rx', 'MarkerSize', 12, 'LineWidth', 2);
        legend({'EEF','Start','Goal'}, 'Location','best');
    end

    % --------- Figure 2: Time series (q/qd/qdd/dist)
    n = double(getfield_or(S, 'n', 2));

    figure('Name','Robot2D Compare: Time Series','Color','w');
    tiledlayout(4, max(1,n), 'Padding','compact', 'TileSpacing','compact');

    for j = 1:max(1,n)
        nexttile; hold on; grid on; title(sprintf('q[%d]', j-1)); xlabel('t (s)');
        for i = 1:numel(names)
            nm = names{i};
            t = getfield_or(S, nm + "_t", zeros(0,1));
            q = getfield_or(S, nm + "_q", zeros(0,n));
            if ~isempty(t) && size(q,1) == size(t,1)
                plot(t, q(:,j), styles(nm), 'Color', colors(nm), 'LineWidth', 1.6);
            end
        end
        if j == 1, legend(names, 'Location','best'); end
    end

    for j = 1:max(1,n)
        nexttile; hold on; grid on; title(sprintf('qd[%d]', j-1)); xlabel('t (s)');
        for i = 1:numel(names)
            nm = names{i};
            t = getfield_or(S, nm + "_t", zeros(0,1));
            qd = getfield_or(S, nm + "_qd", zeros(0,n));
            if ~isempty(t) && size(qd,1) == size(t,1)
                plot(t, qd(:,j), styles(nm), 'Color', colors(nm), 'LineWidth', 1.6);
            end
        end
        if j == 1, legend(names, 'Location','best'); end
    end

    for j = 1:max(1,n)
        nexttile; hold on; grid on; title(sprintf('qdd[%d]', j-1)); xlabel('t (s)');
        for i = 1:numel(names)
            nm = names{i};
            t = getfield_or(S, nm + "_t", zeros(0,1));
            qdd = getfield_or(S, nm + "_qdd", zeros(0,n));
            if ~isempty(t) && size(qdd,1) == size(t,1)
                plot(t, qdd(:,j), styles(nm), 'Color', colors(nm), 'LineWidth', 1.6);
            end
        end
        if j == 1, legend(names, 'Location','best'); end
    end

    % dist in first tile of last row; turn off the rest
    nexttile; hold on; grid on; title('||eef-goal||'); xlabel('t (s)');
    for i = 1:numel(names)
        nm = names{i};
        t = getfield_or(S, nm + "_t", zeros(0,1));
        d = getfield_or(S, nm + "_dist", zeros(0,1));
        if ~isempty(t) && size(d,1) == size(t,1)
            plot(t, d, styles(nm), 'Color', colors(nm), 'LineWidth', 2.0);
        end
    end
    legend(names, 'Location','best');

    for j = 2:max(1,n)
        nexttile; axis off;
    end
end

function pts = fk_2link(q1, q2, l1, l2)
    % Returns 3x2 points: base -> joint1 -> eef
    th1 = double(q1);
    th12 = double(q1 + q2);
    p0 = [0, 0];
    p1 = [l1*cos(th1), l1*sin(th1)];
    p2 = p1 + [l2*cos(th12), l2*sin(th12)];
    pts = [p0; p1; p2];
end

function c2 = lighten_color(c, amount)
    % amount in [0,1]; 0 -> unchanged, 1 -> white
    a = max(0, min(1, double(amount)));
    c = double(c(:)');
    if numel(c) ~= 3
        c = [0 0 0];
    end
    c2 = (1 - a) * c + a * [1 1 1];
end

function v = getfield_or(S, name, defaultValue)
    if isfield(S, char(name))
        v = S.(char(name));
    else
        v = defaultValue;
    end
end
