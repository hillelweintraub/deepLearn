%script for comparing effect of L2 Regularization

basedir = '/home/hweintraub/codeDev/deepLearn/experiments/L2_reg_search/'
taildirs = {'1e-6', '1e-5', '5e-5'}
for ii = 1:length(taildirs)
  file = [taildirs{ii} '/' 'train_stats.mat'];
  load([basedir file])
  train_loss(:,ii) = training_NLL;
  val_loss(:,ii)   = validation_NLL;  
  legend_cell{ii} = taildirs{ii};
end

epochs = 1:length(training_NLL)';

figure()
semilogy(epochs,val_loss)  
title('Validation Negative Log Likelihood')
xlabel('Epoch'), ylabel('NLL'), grid on
%ylim([1e-4 1])
legend(legend_cell)
  