classdef InfoTheo
    % Information-Theoretic measures

    methods (Static)

        function [H, p] = Entropy(arg, options)
            % Entropy of category-like values, or of their distribution
            % SYNTAX:
            % Entropy(x) - where x is category-like vector (categories/strings/integers)
            % Entropy(p) - wehre p is a distribution vector, sums to 1
            % Entropy(..,norm=true) - Normalize by max entropy

            arguments
                arg
                options.norm = false
            end

            arg = arg(:);

            if iscategorical(arg) || all(round(arg) == arg)
                p = groupcounts(arg) / length(arg);
            else
                p = arg(arg>0);
                sm = sum(p);
                assert(abs(1-sm)<eps*numel(p), "Distribution must sum to 1");
                p = p/sm;
            end

            H = -p'*log2(p);
            if options.norm
                H = H/log2(numel(p));
            end

        end

        function [I,H,Hj] = MI(jd)
            % Mutual Information / Total Correlation
            % SYNTAX:
            % MI(X)     where X is a matrix: total correlation between columns X(:,1),X(:,2),..
            % MI(JDObj) where JDObj is a JointDisrib object: total correlation of distribution
            %
            % OUTPUT:
            %   I - mutual information / total correlation
            %   H - vector. H(i) is the entropy of the i-th variable.
            %   Hj - scalar. the joint entropy of all variables

            if ~isa(jd,'JointDistrib')
                jd = JointDistrib(jd);
            end

            H = nan(size(jd.marginals));
            for i = 1 : length(jd.marginals)
                H(i) = InfoTheo.Entropy(jd.marginals{i});
            end
            Hj = InfoTheo.Entropy(jd.joint(:));
            I = sum(H) - Hj;

        end


        function kl = KLDiv(P,Q,options)
            % KL-Divergence
            % P,Q - matrices s.t. each row is a distribution (sums to 1)
            %
            % kl = KLDiv(P,Q) returns a size(P,1)*size(Q,1) matrix, such
            %   that kl(i,j) is the KLDiv between P(i,:) & Q(j,:)
            %
            % kl = KLDiv(P,Q,rows=true) returns a vector such that kl(i) is
            %   the KLDiv between P(i,:) & Q(i,:). P & Q must be the same size.

            arguments
                P (:,:) double
                Q (:,:) double
                options.rows (1,1) logical = false
            end

            assert(size(P,2)==size(Q,2));

            P(P==0) = eps;
            Q(Q==0) = eps;

            if options.rows
                assert(size(P,1)==size(Q,1));
                kl = nan([size(P,1),1]);
                for i = 1 : size(P,1)
                    kl(i) = P(i,:)*log2(P(i,:)./Q(i,:))' ;
                end
            else
                kl = nan(size(P,1),size(Q,1));
                for i = 1 : size(P,1)
                    for j = 1 : size(Q,1)
                        kl(i,j) = P(i,:)*log2(P(i,:)./Q(j,:))' ;
                    end
                end
            end

        end

        function js = JSDiv(P,Q,options)
            % Jensen–Shannon divergence between distributions P(i,:) & Q(j,:)
            %
            % P,Q - matrices s.t. each row is a distribution (sums to 1)
            %
            % js = JSDiv(P,Q) returns a size(P,1)*size(Q,1) matrix, such
            %   that js(i,j) is the JSDiv between P(i,:) & Q(j,:)
            %
            % js = JSDiv(P,Q,rows=true) returns a vector such that js(i) is
            %   the JSDiv between P(i,:) & Q(i,:). P & Q must be the same size.
            arguments
                P (:,:) double
                Q (:,:) double
                options.rows (1,1) logical = false
            end
            options = namedargs2cell(options);
            js = .5 * (InfoTheo.KLDiv(P,Q,options{:}) + InfoTheo.KLDiv(Q,P,options{:}));
        end


        function c1 = EntropyBias(N,K)
            % The first correction term for an entropy that was computed over N samples, for a domain size K.
            % This is the bias that results from estimating a probability P(k), {k=1,2,..K},
            % using a histogram count over N samples.
            %
            % SYNTAX:
            %   entropyBias(x) the bias for an array x, from which a probability (and entropy) was estimated.
            %   entropyBias(N,K)
            %       N - sample size. i.e. N = length(x)
            %       K - domain size. i.e. K = number of possible x values
            %
            % OUTPUT:
            %   c1. The first correction term, i.e.: H_true = H_naive + c1 (+c2,..)

            if nargin==1
                K = length(unique(N));
                N = length(N);
            end
            c1 = .5*(K-1)/N;
        end

    end

end