function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

% F1, PREC, recall formules are simple ratios. 
    tp = 1
    fn = 1
    fp = 1
    
    prediction = (pval < epsilon); 
%{
tp is the number of true positives: the ground truth label says it's an
anomaly and our algorithm correctly classied it as an anomaly.
� fp is the number of false positives: the ground truth label says it's not
an anomaly, but our algorithm incorrectly classied it as an anomaly.
� fn is the number of false negatives: the ground truth label says it's an
anomaly, but our algorithm incorrectly classied it as not being anoma-
lous.
%}

    tp = sum((prediction == 1) &(yval == 1))    % case is 1 and pred is 1
    tn = sum((prediction == 0) & (yval ==0))    % case is 0 and pred also is 0
    fp = sum((prediction ==1) & (yval==0))  % pred is 1 while acutall it is not 
    fn = sum((prediction == 0)& (yval ==1)) % pred is -ve while actualy it was 1  
    
    recall = tp/(tp+fn)
    prec =tp/(tp + fp)

    F1 = 2 *prec*recall/(prec + recall)




    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
