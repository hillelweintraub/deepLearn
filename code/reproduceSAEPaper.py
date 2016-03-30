#put a comment here

import os
import sys
import timeit
import cPickle
import shutil

import scipy.io as sio
import numpy

import theano
import theano.tensor as T

from tutorialCode.SdA import SdA
from util.generateDataSets import split_to_train_val_test
from util.logger import Logger


#theano.config.optimizer = 'None'
#theano.config.exception_verbosity = 'high'

def load_data(dataset,split_proportions,logger,shared=True):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset 
    '''

    #############
    # LOAD DATA #
    #############

    # Download the dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "Data",
            dataset
        )
        if os.path.isfile(new_path):
            dataset = new_path   
        else:
	    raise RuntimeError("could not locate dataset file: %s" % dataset)

    print '... loading data'

    # Load the dataset    
    train_set, valid_set, test_set = split_to_train_val_test(dataset,split_proportions,
							     random_split=False)
    
    #log the dataset statistics
    train_y, valid_y, test_y = train_set[1], valid_set[1], test_set[1]
    logger.log("Dataset statistics...")
    for i in xrange( len(numpy.unique(train_y)) ):
      logger.log("class: %d train: %d valid: %d test: %d Total: %d" % (
	  i,
	  sum(train_y==i),
	  sum(valid_y==i),
	  sum(test_y==i),
	  sum(train_y==i)+sum(valid_y==i)+sum(test_y==i)
	)
      )
    
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets us get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    if shared:
      test_set_x, test_set_y = shared_dataset(test_set)
      valid_set_x, valid_set_y = shared_dataset(valid_set)
      train_set_x, train_set_y = shared_dataset(train_set)
      
      rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
              (test_set_x, test_set_y)]
    else:
      rval = [train_set, valid_set, test_set]
    return rval

def run_SAE_experiment(pretrain_lr=0.1,pretraining_epochs=3300,
		       finetune_lr=0.1,training_epochs=4e5,
		       L1_reg=0.0,L2_reg=1e-4,
		       dataset='KSC.pkl',
		       split_proportions = [6,2,2],
		       hidden_layers_sizes=[20],
		       corruption_levels=[0.],
		       batch_size=20,
		       log_file='log',
		       restart = False,
		       use_rate_schedule=True,
		       load_pretrained_weights=False):
    """
    Reproduce the paper...
    """
    assert not(restart and load_pretrained_weights)
    assert not(load_pretrained_weights and len(hidden_layers_sizes)!=5)
    assert len(hidden_layers_sizes)==len(corruption_levels), \
           "Error: hidden_layers_sizes and corruption_levels need to be of equal length"
	 
    pretrain_rate_decay = (type(pretrain_lr)==tuple)
    train_rate_decay = (type(finetune_lr)==tuple)
    assert pretrain_rate_decay or type(pretrain_lr)==float
    assert train_rate_decay or type(finetune_lr)==float
    assert not (use_rate_schedule and train_rate_decay), ('Error:',
      'Can not use adaptive rate schedule and linear rate schedule together' )
    
    #cast number of epochsto int
    pretraining_epochs = int(pretraining_epochs)
    training_epochs = int(training_epochs)
    
    #check for linear rate schedules
    if pretrain_rate_decay:
      linear_pretrain_rates = True
      pretrain_rates = numpy.linspace(pretrain_lr[0],pretrain_lr[1],pretraining_epochs)
    else:
      pretrain_rates = [pretrain_lr]*pretraining_epochs
    
    if train_rate_decay:
      linear_train_rates = True
      train_rates = numpy.linspace(finetune_lr[0],finetune_lr[1],training_epochs)
    else:
      train_rates = [finetune_lr]*training_epochs
	 
    #create a log object
    logger = Logger(log_file)
    
    #log run params
    if restart: logger.log("Restarting run using old best_model")
    logger.log("Running SAE Experiment...")
    logger.add_newline()
    logger.log("Runtime params:")
    logger.log("pretrain_lr=%s" % str(pretrain_lr))
    logger.log("pretraining_epochs=%d" % pretraining_epochs)
    logger.log("finetune_lr=%s" % str(finetune_lr))
    logger.log("training_epochs=%d" % training_epochs)
    logger.log("L1_reg=%f" % L1_reg)
    logger.log("L2_reg=%f" % L2_reg)
    logger.log("dataset=%s" % dataset)
    logger.log("split_proportions=%s" % str(split_proportions))
    logger.log("hidden_layers_sizes=%s" % str(hidden_layers_sizes))
    logger.log("corruption_levels=%s" % str(corruption_levels))
    logger.log("batch_size=%d" % batch_size)
    logger.log("use_rate_schedule=%s" % use_rate_schedule)
    logger.log("load_pretrained_weights=%s" % load_pretrained_weights)
    logger.add_newline()
    
    
    datasets = load_data(dataset,split_proportions,logger)
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]    
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    
    # numpy random generator    
    numpy_rng = numpy.random.RandomState(89677)
    logger.log( '... building the model')
    # construct the stacked denoising autoencoder class    
    #since labels were cast to int32 need to do this to get the shared variable 
    shared_train_set_y = train_set_y.owner.inputs[0]   
    if not restart:
      sda = SdA(
          numpy_rng=numpy_rng,
          n_ins=train_set_x.get_value(borrow=True).shape[1],
          hidden_layers_sizes=hidden_layers_sizes,
          n_outs=numpy.unique(shared_train_set_y.get_value(borrow=True)).size,
          L1_reg=L1_reg,
          L2_reg=L2_reg
      )
    elif restart:
      logger.log("loading model from best_model.pkl")
      sda = cPickle.load(open('best_model.pkl','r'))
    elif load_pretrained_weights:
      logger.log("loading model from pretrained_model.pkl")
      sda = cPickle.load(open('pretrained_model.pkl','r'))

    #create dictionary to store training stat accumulation arrays for easy pickling
    train_stat_dict = {}

    #########################
    # PRETRAINING THE MODEL #
    #########################
    pretrainig_costs = [ [] for i in xrange(sda.n_layers) ] # average pretrainig cost at each epoch
    train_stat_dict['pretrainig_costs'] = pretrainig_costs
    if not (restart or load_pretrained_weights or SKIP_PRETRAINING):
      logger.log( '... getting the pretraining functions')
      pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                  batch_size=batch_size)
      
      logger.log( '... pre-training the model')
      start_time = timeit.default_timer()
      
      ## Pre-train layer-wise    
      for i in xrange(sda.n_layers):
          # go through pretraining epochs
          for epoch in xrange(pretraining_epochs):
              # go through the training set
              c = []
              for batch_index in xrange(n_train_batches):
                  c.append(pretraining_fns[i](index=batch_index,
                           corruption=corruption_levels[i],
                           lr=pretrain_rates[epoch]))
              logger.log('Pre-training layer %i, epoch %d, cost ' % (i, epoch) )
              logger.log( str(numpy.mean(c)) )
              pretrainig_costs[i].append(numpy.mean(c))
      
      end_time = timeit.default_timer()
      
      #save the pretrained model
      with open('pretrained_model.pkl', 'w') as f:
          cPickle.dump(sda, f)
      
      logger.log( 'The pretraining code for file ' +
                            os.path.split(__file__)[1] +
                            ' ran for %.2fm' % ((end_time - start_time) / 60.)
      )
    else:
      logger.log("skipping pretraining")
    
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    logger.log( '... getting the finetuning functions')
    ( train_fn, validate_model_NLL, 
      validate_model_zero_one, test_model ) = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size
        )

    logger.log( '... finetunning the model')
    # early-stopping parameters
    patience = 100 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0    
    iter = 0 # global minibatch iteration
    minibatch_avg_NLL = [] #array to accumulate NLL cost over over minibatches
    training_NLL = []     # average training NLL cost at each epoch (really after val_freq iters)
    validation_NLL = []  # average validation NLL cost at each epoch
    validation_zero_one = [] # average zero one cost at each epoch (% misclassified)
    train_stat_dict['training_NLL'] = training_NLL
    train_stat_dict['validation_NLL'] = validation_NLL
    train_stat_dict['validation_zero_one'] = validation_zero_one    
    while (epoch < training_epochs) and (not done_looping):        
        epoch = epoch + 1                
        for minibatch_index in xrange(n_train_batches):
	    iter += 1
            minibatch_avg_NLL.append( train_fn(minibatch_index,lr= train_rates[epoch-1] ) )

            if iter % validation_frequency == 0:	
              """validation zero one loss """	      
              validation_zero_one_losses = validate_model_zero_one()
              validation_zero_one.append( numpy.mean(validation_zero_one_losses) )              
                  
	      #validation NLL cost
	      validation_NLL_losses = validate_model_NLL()
	      validation_NLL.append( numpy.mean(validation_NLL_losses) )
	      
	      #training NLL cost
	      training_NLL.append( numpy.mean(minibatch_avg_NLL) )
	      minibatch_avg_NLL = [] #reset the NLL accumulator
	      
	      logger.log( 'epoch %i, minibatch %i/%i:' % (epoch, minibatch_index + 1, n_train_batches))
	      logger.log('\ttraining NLL loss: %f ' % training_NLL[-1])
	      logger.log('\tvalidation NLL loss: %f ' % validation_NLL[-1])
	      logger.log('\tvalidation zero one loss: %f %%' % (validation_zero_one[-1] * 100.))

              # if we got the best validation score until now
              if validation_zero_one[-1] < best_validation_loss: 		  
                                    
                  #improve patience if loss improvement is good enough
                  if (
                      validation_zero_one[-1] < best_validation_loss *
                      improvement_threshold
                  ):
                      patience = max(patience, iter * patience_increase)
                  else:
		      print "improvemnt not good enough: %f" % (validation_zero_one[-1]/best_validation_loss)
                      
                  # save best validation score and iteration number
                  best_validation_loss = validation_zero_one[-1]
                  best_iter = iter

                  # test it on the test set
                  test_zero_one_losses = test_model()
                  test_score = numpy.mean(test_zero_one_losses)
                  print '\t\ttest zero one loss of best model %f %%' % (test_score * 100.)
		
	          #save the best model
	          with open('best_model.pkl', 'w') as f:
                      cPickle.dump(sda, f)
	      
            if patience <= iter:
	        pass
                #done_looping = True
                #break
        if use_rate_schedule and epoch%100==0:
	      if validation_NLL[epoch-100]-validation_NLL[epoch-1]<1e-4:	
		finetune_lr = max(finetune_lr/2.,1e-6)
		train_rates = [finetune_lr]*training_epochs
		logger.log("Reducing learning rate. new rate: %f" % finetune_lr)	          
        
    #save train_stat_dict to a .mat file
    sio.savemat('train_stats.mat',train_stat_dict)
    #with open('train_stat_dict.pkl','w') as f:
    #  cPickle.dump(train_stat_dict,f)
    end_time = timeit.default_timer()
    logger.log(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter, test_score * 100.)
    )
    logger.log ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.)
    )

    logger.close()
  
   
  
def grid_search_lr():  
  basedir = '/home/hweintraub/codeDev/deepLearn/experiments/rate_search/'
  if not os.path.exists(basedir): os.makedirs(basedir)
  rates = [1e-1, 1e-2, 5e-3, 1e-3]
  for rate in rates:
    run_SAE_experiment(pretrain_lr=0.6,pretraining_epochs=5000,
		       finetune_lr=rate,training_epochs=5e4,
		       L1_reg=0.0,L2_reg=0.0,
		       dataset='KSC.pkl',
		       split_proportions = [6,2,2],
		       hidden_layers_sizes=[20],
		       corruption_levels=[0.],
		       batch_size=50,
		       log_file = 'log',
		       restart = False)
    copydir = basedir+'lr=%.3f'%rate
    if not os.path.exists(copydir): os.makedirs(copydir)
    shutil.copy('log',copydir)
    shutil.copy('best_model.pkl',copydir)
    shutil.copy('train_stats.mat',copydir)

def grid_search_depth():  
  basedir = '/home/hweintraub/codeDev/deepLearn/experiments/depth_search/no_pretrain/'
  if not os.path.exists(basedir): os.makedirs(basedir)
  depths = [5,4,3,2,1]
  for depth in depths:
    run_SAE_experiment(pretrain_lr=0.6,pretraining_epochs=5000,
		       finetune_lr=0.05,training_epochs=5e4,
		       L1_reg=0.0,L2_reg=0.0,
		       dataset='KSC.pkl',
		       split_proportions = [6,2,2],
		       hidden_layers_sizes=[20]*depth,
		       corruption_levels=[0.]*depth,
		       batch_size=64,
		       log_file = 'log',
		       restart = False,
		       use_rate_schedule=True)
    copydir = basedir+'depth=%d'%depth
    if not os.path.exists(copydir): os.makedirs(copydir)
    shutil.copy('log',copydir)
    shutil.copy('best_model.pkl',copydir)
    shutil.copy('train_stats.mat',copydir)

def grid_search_l2_reg():
  basedir = '/home/hweintraub/codeDev/deepLearn/experiments/L2_reg_search/'
  if not os.path.exists(basedir): os.makedirs(basedir)
  exponents = [5,4]
  depth=5
  for exp in exponents:
    L2_reg = float('5e-%d'%exp)
    run_SAE_experiment(pretrain_lr=0.6,pretraining_epochs=5000,
		       finetune_lr=0.05,training_epochs=5e4,
		       L1_reg=0.0,L2_reg=L2_reg,
		       dataset='KSC.pkl',
		       split_proportions = [6,2,2],
		       hidden_layers_sizes=[20]*depth,
		       corruption_levels=[0.]*depth,
		       batch_size=64,
		       log_file = 'log',
		       restart = False,
		       use_rate_schedule=False)
    copydir = basedir+'5e-%d'%exp
    if not os.path.exists(copydir): os.makedirs(copydir)
    shutil.copy('log',copydir)
    shutil.copy('best_model.pkl',copydir)
    shutil.copy('train_stats.mat',copydir)
  
if __name__ == '__main__':
  SKIP_PRETRAINING = True
  #depth = 5
  #run_SAE_experiment(pretrain_lr=0.6,pretraining_epochs=5000,
		       #finetune_lr=0.05,training_epochs=4e5,
		       #L1_reg=0.0,L2_reg=5e-5,
		       #dataset='KSC.pkl',
		       #split_proportions = [6,2,2],
		       #hidden_layers_sizes=[20]*depth,
		       #corruption_levels=[0.]*depth,
		       #batch_size=64,
		       #log_file = 'log',
		       #restart = False,
		       #use_rate_schedule=False,
		       #load_pretrained_weights = True)
  #grid_search_lr()   	
  grid_search_depth()
  #grid_search_l2_reg()
 
 
 

 
 
 
 
 
 
 
 
 
 
 
  
  
  
  