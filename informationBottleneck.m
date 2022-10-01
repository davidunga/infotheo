function [p_t_given_x,info] = informationBottleneck(px,pxy,beta,nClusters,maxItr,epsln)
% Iterative Information Bottleneck.
% Given disributions of two variables X,Y, Information Bottleneck constructs
% a new variable T, where T is a compression of X which preseves information about Y.
% That is- T tells us as much as possible about Y, but as little as possible about X.
% Formally, this algorithm finds the conditional disribution P(T|X) that minimizes:
%   I(X;T) - beta*I(Y;T)
% Where
%   I(A;B) is the mutual information between variables A and B.
%   beta is a parameter that determines the trade-off between
%   compression and reliability. Higher beta --> higher reliability.
%
% INPUT:
%   px - vector representing marginal X probability P(x)
%   pxy - matrix representing joint X,Y probability P(X,Y) such that:
%       pxy(i,j) <--> P( X==x_i and Y==y_j )
%   beta - inverse temprature parameter
%   nClusters - number of compressed variables (clusters)
%
% OPTIONAL INPUT:
%   maxItr - maximum number of iterations. default = 10^3.
%   epsln - stop when the update of P(T|X) is less than epsln. default =  10^-14.
%
% OUTPUT:
%   p_t_given_x - P(T|X), soft compression mapping X->T
%   info - struct of information about clustering performance

% .........................................................................
% INITIALIZE

% --- Parameters and constants:

if ~exist('maxItr','var'), maxItr = 10^3 ; end % max number of iterations
if ~exist('epsln','var'), epsln = 10^-14 ; end % stopping condition

Nx = size(pxy,1); 	% number of X values
Ny = size(pxy,2); 	% number of Y values
Nt = nClusters; 	% number of T values

% --- Sort input data:

px = px(:);
p_y_given_x = pxy ./ repmat( px , 1, Ny ) ;


% --- Generate initial disributions according to a random initial P(t|x):

% Random P(t|x):
p_t_given_x = rand(Nx,Nt);
p_t_given_x = p_t_given_x ./ repmat( sum(p_t_given_x,2) , 1, Nt ) ;

% P(t) = sum_x{ P(t|x)P(x) } :
p_t = p_t_given_x'*px ;
p_t = p_t ./ sum(p_t); % (to avoid numerical issues)

% P(y|t) = 1/P(t) * sum_x{ P(x,y)P(t|x) } :
p_y_given_t = (pxy'*p_t_given_x)' ./ repmat( p_t , 1 , Ny );


% .........................................................................
% CORE

jsdist = ones(1,Nx) + epsln ;
itr = 0;
while any(jsdist > epsln) && itr < maxItr
    % While update of P(t|x) is still significant
    
    itr = itr+1; % iteration counter
    
    % Save current P(t|x) to be compared to P(t|x) after iteration:
    p_t_given_x_prev = p_t_given_x ;
    
    % ---------------------------------------------------------------------
    % Compute new estimates:
    
    % KL-Dist between P(y|x) and P(y|t) :
    kldist = InfoTheo.KLDiv(p_y_given_x,p_y_given_t);
    
    % Update P(t|x) according to P(t|x) = P(t)*exp(-beta*kldist) :
    p_t_given_x = repmat( p_t', Nx , 1 ) .* exp(-beta*kldist);
    p_t_given_x = p_t_given_x ./ repmat( sum(p_t_given_x,2) , 1, Nt ) ;
    
    % P(t):
    p_t = p_t_given_x'*px ;
    p_t = p_t ./ sum(p_t); % (re-normalize to avoid numerical issues)
    
    % P(y|t):
    p_y_given_t = (pxy'*p_t_given_x)' ./ repmat( p_t , 1 , Ny );
    
    % ---------------------------------------------------------------------
    % Test convergence:
    
    % Compare P(t|x) before and after this iteration took place:
    jsdist = InfoTheo.JSDiv(p_t_given_x,p_t_given_x_prev,rows=true);
    
end

% Rename clusters according to their size, cluster 1 is the largest:
p_t_given_x = sortClustersBySize(p_t_given_x);

info = struct(itr=itr, jsdist=jsdist, converge=jsdist<epsln);

end % End main function


function p_t_given_x = sortClustersBySize(p_t_given_x)
% Re-enumerate clusters s.t they are sorted by size in descending order
nClusters = size(p_t_given_x,2) ;
[~,mxj] = max( p_t_given_x , [] , 2 ) ;
clusterSize = hist(mxj,1:nClusters);
[~,si] = sort(clusterSize,'descend');
p_t_given_x = p_t_given_x(:,si);
end





