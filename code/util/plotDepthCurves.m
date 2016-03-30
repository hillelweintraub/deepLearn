%script for plotting comparison of different depth nets

basedir = '/home/hweintraub/codeDev/deepLearn/experiments/depth_search/'
colors = 'rgbkm'  
for depth = 1:5
  file = sprintf('depth=%d/train_stats.mat',depth');
  load([basedir file])
  train_loss(:,depth) = training_NLL;
  val_loss(:,depth)   = validation_NLL;  
  legend_cell{depth} = sprintf('depth=%d',depth);
end

epochs = 1:length(training_NLL)';

subplot(121)
semilogy(epochs,train_loss)  
title('Training Negative Log Likelihood')
xlabel('Epoch'), ylabel('NLL'), grid on
%ylim([1e-4 1])
legend(legend_cell)

subplot(122)
semilogy(epochs,val_loss)  
title('Validation Negative Log Likelihood')
xlabel('Epoch'), ylabel('NLL'), grid on
%ylim([1e-4 1])
legend(legend_cell)

