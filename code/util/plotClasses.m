%function to plot what the different classes look like
%from a hyperspectral image dataset

function [examplesPerClass] = plotClasses(data,data_gt)
%Inputs:
% data - A hyperspectral data cube with the first two dimensions
%        being spatial and the third being spectral
% data_gt - A 2d class map containig a label for every spatial
%           pixel in data

%Outputs:
% examplesPerClass - the number of examples in each class
%                    i.e. examplesPerClass(i) is the number of 
%                    examples with class label i


%reshape the data for plotting purposes
[m n k] = size(data)
data = reshape(shiftdim(data,2),k,[]);
data_gt = data_gt(:);

%plot the classes
labels = unique(data_gt);
labels(labels==0)=[]; %do not include 0 in the set of class labels
numClasses = length(labels);
examplesPerClass=zeros(numClasses,1);
for ii=1:numClasses
  examplesPerClass(ii) = sum(data_gt==labels(ii));
end

numExamples = 100; %the number of examples per class to plot
for ii=1:numClasses
  subplot(3,ceil(numClasses/3),ii)
  exampleIndices = find(data_gt==labels(ii))(randperm(examplesPerClass(ii),numExamples));
  plot( log10(data(:,exampleIndices)) )
  %ylim([0 500])
end
