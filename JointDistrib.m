classdef JointDistrib
    % Handle frequency-based joint distribution
    
    properties
        domains = {};   % domains{i} = vector of possible values of i-th variable
        joint = [];     % joint distribution matrix, size = length(domains{1}) x length(domains{2}) x ..
        marginals = {}; % marginals{i} = marginal distribution of i-th variable
        n = 0;          % number of samples
    end

    methods

        function obj = JointDistrib(X, domains)
            % X - N (samples) x M (variables) array of category-like values (integers/categorical/strings)
            % domains - cell vector with length=size(X,2), such that
            %   domains{j} is a vector of the possible values X(:,j) can
            %   obtain. default: domains{j} = unique(X(:,j)).

            if islogical(X)
                X = double(X);
            end

            num_vars = size(X,2);
            if ~exist('domains','var') || isempty(domains)
                obj.domains = repmat({{}},[1,num_vars]);
            else
                obj.domains = domains(:)';
            end

            for j = 1 : num_vars
                if isempty(obj.domains{j})
                    [obj.domains{j},~,X(:,j)] = unique(X(:,j));
                    obj.domains{j} = obj.domains{j}';
                else
                    [~,X(:,j)] = ismember(X(:,j),obj.domains{j});
                end
            end

            open_rows = 1:size(X, 1);
            joint_counts = zeros(cellfun(@length, obj.domains));
            while ~isempty(open_rows)
                ii = all(X(open_rows, :) == X(open_rows(1), :), 2);
                ixs = num2cell(X(open_rows(1), :));
                joint_counts(sub2ind(size(joint_counts), ixs{:})) = sum(ii);
                open_rows = open_rows(~ii);
            end
            
            obj.n = sum(joint_counts(:));
            obj.joint = joint_counts / obj.n;
            obj.marginals = cell(1,num_vars);
            for i = 1 : num_vars
                obj.marginals{i} = reshape(sum(obj.joint, [1:(i-1),(i+1):num_vars]), size(obj.domains{j}));
            end

        end

        function c = joint_counts(obj)
            c = obj.joint * obj.n;
        end

        function c = maringal_counts(obj)
            c = cellfun(@(x) x * obj.n, obj.marginals, 'UniformOutput',false);
        end

    end

end
