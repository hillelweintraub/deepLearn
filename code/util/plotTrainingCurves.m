%script for plotting how training progressed

%assumes data was already loaded

%pretrain_epochs = 1:length(pretrainig_costs);
epochs = 1:length(training_NLL);

%  subplot(311)
%  semilogy(pretrain_epochs',pretrainig_costs')
%  title('Preraining Loss'),xlabel('Epoch'),ylabel('Cross-entropy')
%  grid on

%subplot(312)
figure()
loglog(epochs,training_NLL,'r',epochs, validation_NLL,'b')
legend('train\_nll','valid\_nll')
title('Negative Log Likelihood'), xlabel('Epoch'),ylabel('NLL')
grid on

%subplot(313)
figure()
loglog(epochs,validation_zero_one,'b')
title('Zero One Loss'), xlabel('Epoch'),ylabel('zero one loss')
grid on
