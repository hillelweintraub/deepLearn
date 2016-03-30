#use lasagne to build some nets

import cPickle
import timeit
import os
import shutil

import numpy as np
import scipy.io as sio

import lasagne
import theano
import theano.tensor as T

from util.logger import Logger
from reproduceSAEPaper import load_data



def build_network(input_var,
		  input_size,
		  hidden_sizes,
		  hidden_nonlinearity,
		  dropout_probs,
		  output_size):
 
  #validate inputs 
  assert len(hidden_sizes)==len(dropout_probs)
  #Input layer
  net = lasagne.layers.InputLayer(shape=(None,input_size),
				  input_var=input_var)
  #dropout at the input
  #net = lasagne.layers.dropout(net, p=0.2)
  #Hidden layers
  depth = len(hidden_sizes)
  for i in xrange(depth):
    net = lasagne.layers.DenseLayer(net,
				    hidden_sizes[i],
				    nonlinearity=hidden_nonlinearity)
    if dropout_probs[i]>0:
      net = lasagne.layers.dropout(net,
				   p=dropout_probs[i])
  #Output layer
  net = lasagne.layers.DenseLayer(net,
				  output_size,
				  nonlinearity=lasagne.nonlinearities.softmax)
  return net 
				    
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
  assert len(inputs) == len(targets)
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0,len(inputs)-batchsize+1,batchsize):
    if shuffle:
      excerpt = indices[start_idx:start_idx+batchsize]
    else:
      excerpt = slice(start_idx,start_idx + batchsize)
    yield inputs[excerpt], targets[excerpt]
  

def run_deepNet(dataset = 'KSC.pkl',
		split_proportions = [6,2,2],                
		hidden_sizes = [20],
		hidden_nonlinearity = lasagne.nonlinearities.rectify,
		dropout_probs = [0.5],
		learning_rate = 0.1,
		momentum = 0.9,
		num_epochs = int(5e4),
		minibatch_size = 64,
		log_file = 'log'):
  
  #create a log object
  logger = Logger(log_file)
  
  #log run params
  logger.log("Running run_deepNet Experiment...")
  logger.add_newline()
  logger.log("Runtime params:")  
  logger.log("dataset=%s" % dataset)
  logger.log("split_proportions=%s" % str(split_proportions))
  logger.log("hidden_sizes=%s" % str(hidden_sizes))
  logger.log("hidden_nonlinearity=%s" % str(hidden_nonlinearity))
  logger.log("dropout_probs=%s" % str(dropout_probs))
  logger.log("learning_rate=%s" % str(learning_rate))
  logger.log("momentum=%s" % str(momentum))
  logger.log("num_epochs=%d" % num_epochs)
  logger.log("minibatch_size=%d" % minibatch_size)  
    
  #Load the data
  train_set,val_set,test_set = load_data(dataset,split_proportions,
					    logger,shared=False)
  x_train, y_train = train_set
  x_val, y_val = val_set
  x_test, y_test = test_set
  #normalize data to zero mean unit variance
  x_mean = np.mean(x_train)
  x_std = np.std(x_train)
  x_train = (x_train-x_mean)/x_std
  x_val = (x_val-x_mean)/x_std
  x_test = (x_test-x_mean)/x_std
      
  #prepare theano variables for inputs and targets
  input_var = T.matrix('inputs')
  target_var = T.ivector('targets')
  #build the model
  logger.log( '... building the model')
  input_size = x_train.shape[1]
  output_size=np.unique(y_train).size
  #net = build_network(input_var,
		      #input_size,
		      #hidden_sizes,
		      #hidden_nonlinearity,
		      #dropout_probs,
		      #output_size)
  net = cPickle.load(open('best_model.pkl','r'))
  layers = lasagne.layers.get_all_layers(net)
  input_var = layers[0].input_var
  #create loss expression for training
  logger.log( '... building expressions and compiling train functions')
  predicition = lasagne.layers.get_output(net)
  loss = lasagne.objectives.categorical_crossentropy(predicition,target_var)
  loss = loss.mean()
  #create update expressions for training
  params = lasagne.layers.get_all_params(net,trainable=True)
  #linearly decay learning rate
  learning_rate = np.linspace(learning_rate[0],learning_rate[1],num_epochs)
  lr = theano.shared(np.array(learning_rate[0],dtype=theano.config.floatX))
  #linearly grow momentum
  momentum = np.linspace(momentum[0],momentum[1],num_epochs)
  mom = theano.shared(np.array(momentum[0],dtype=theano.config.floatX))
  updates = lasagne.updates.nesterov_momentum(loss,params,
					      learning_rate=lr,
					      momentum=mom)
  #create loss expression for validation/testing
  test_prediction = lasagne.layers.get_output(net,deterministic=True)
  test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
							  target_var)
  test_loss = loss.mean()
  #create an expression for the classification accuracy
  test_acc = T.mean(T.eq(T.argmax(test_prediction,axis=1), target_var),
		    dtype=theano.config.floatX)
  #compile the training function
  train_fn = theano.function([input_var,target_var],
			     loss,
			     updates=updates)
  #compile a validation function for the validation loss and accuracy
  val_fn = theano.function([input_var,target_var],
			   [test_loss,test_acc])
  #train the model
  logger.log( '... training the model')
  start_time = timeit.default_timer()
  best_validation_loss = np.inf    
  training_NLL = []     # average training NLL cost at each epoch (really after val_freq iters)
  validation_NLL = []  # average validation NLL cost at each epoch
  validation_zero_one = [] # average zero one cost at each epoch (% misclassified)
  train_stat_dict = {}
  train_stat_dict['training_NLL'] = training_NLL
  train_stat_dict['validation_NLL'] = validation_NLL
  train_stat_dict['validation_zero_one'] = validation_zero_one    
    
  for epoch in xrange(num_epochs):
    #do a pass over the training data
    lr.set_value(learning_rate[epoch])
    mom.set_value(momentum[epoch])
    train_err=0
    train_batches=0
    for batch in iterate_minibatches(x_train,y_train,
				     minibatch_size,shuffle=True):
      inputs, targets = batch
      train_err += train_fn(inputs, targets)
      train_batches += 1
    #do a pass over the validation data
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(x_val, y_val,
				     minibatch_size,shuffle=False):
      inputs, targets = batch
      err, acc = val_fn(inputs, targets)
      val_err += err
      val_acc += acc
      val_batches += 1
    #record results
    training_NLL.append(train_err / train_batches)
    validation_NLL.append(val_err / val_batches)
    validation_zero_one.append(1-(val_acc/val_batches))
    logger.log( 'epoch %i:' % (epoch))
    logger.log('\ttraining NLL loss: %f ' % training_NLL[-1])
    logger.log('\tvalidation NLL loss: %f ' % validation_NLL[-1])
    logger.log('\tvalidation zero one loss: %f %%' % (validation_zero_one[-1] * 100.))
    # if we got the best validation score until now                                        
    if validation_zero_one[-1] < best_validation_loss: 		                            
        # save best validation score and iteration number
        best_validation_loss = validation_zero_one[-1]
        best_epoch = epoch         
        #save the best model
        with open('best_model.pkl', 'w') as f:
                cPickle.dump(net,f)
    # update the best model in a sliding window looking back 50 epochs
    #window_start = max(len(validation_zero_one)-50,0)
    #window_end = len(validation_zero_one)
    #if validation_zero_one[-1] == min(validation_zero_one[window_start:window_end]):
        ## save best validation score and iteration number
        #best_window_validation_loss = validation_zero_one[-1]
        #best_window_epoch = epoch         
        ##save the best model
        #with open('best_window_model.pkl', 'w') as f:
                #cPickle.dump(net,f)
    if (epoch-best_epoch)>1e4: 
      logger.log("Early stopping...")
      break
                 
  ######post training#######
  
  #save train_stat_dict to a .mat file
  sio.savemat('train_stats.mat',train_stat_dict)                        
  #with open('train_stat_dict.pkl','w') as f:                        
  #  cPickle.dump(train_stat_dict,f)     
    
  # After training, we compute and print the test error:
  #load best model
  logger.log("loading model from best_model.pkl")
  net = cPickle.load(open('best_model.pkl','r'))
  #logger.log("loading model from best_window_model.pkl")
  #window_net = cPickle.load(open('best_window_model.pkl','r'))
  test_err, test_acc = predict(net,x_test,y_test)
  test_score = 1 - test_acc
  #test_err, test_acc = predict(window_net,x_test,y_test)
  #window_test_score = 1 - test_acc
  end_time = timeit.default_timer()
  logger.log(
      (
          'Optimization complete with best validation score of %f %%, '
          'on epoch %i, '
          'with test performance %f %%'
      )
      % (best_validation_loss * 100., best_epoch, test_score * 100.)
  )    
  logger.log ('The training code for file ' +
                        os.path.split(__file__)[1] +
                        ' ran for %.2fm' % ((end_time - start_time) / 60.)
  )

  logger.close()
  
def predict(net,x_test,y_test):
  #create loss expression for validation/testing
  layers = lasagne.layers.get_all_layers(net)
  input_var = layers[0].input_var
  target_var = T.ivector('targets') 
  test_prediction = lasagne.layers.get_output(net,deterministic=True)
  test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
							  target_var)
  test_loss = test_loss.mean()
  #create an expression for the classification accuracy
  test_acc = T.mean(T.eq(T.argmax(test_prediction,axis=1), target_var),
		    dtype=theano.config.floatX)  
  #compile a validation function for the validation loss and accuracy
  val_fn = theano.function([input_var,target_var],
			   [test_loss,test_acc])
  err, acc = val_fn(x_test,y_test)
  return err, acc

def grid_search_layer_sizes():  
  basedir = '/home/hweintraub/codeDev/deepLearn/experiments/lasagne/layer_size_search/'
  if not os.path.exists(basedir): os.makedirs(basedir)
  sizes = [100,200,300,400,500]
  for size in sizes:
    depth = 2
    run_deepNet(dataset = 'KSC.pkl',
		split_proportions = [6,2,2],                
		hidden_sizes = [size]*depth,
		hidden_nonlinearity = lasagne.nonlinearities.rectify,
		dropout_probs = [0.5]*depth,
		learning_rate = 0.001,
		momentum = 0.9,
		num_epochs = int(5e4),
		minibatch_size = 64,
		log_file = 'log')
    copydir = basedir+'layer_size=%d'%size
    if not os.path.exists(copydir): os.makedirs(copydir)
    shutil.copy('log',copydir)
    shutil.copy('best_model.pkl',copydir)
    shutil.copy('train_stats.mat',copydir)

if __name__ == '__main__':
  depth = 10
  run_deepNet(dataset = 'KSC.pkl',
		split_proportions = [6,2,2],                
		hidden_sizes = [2e3,2e2]*(depth/2),
		hidden_nonlinearity = lasagne.nonlinearities.rectify,
		dropout_probs = [0.5,0.0]*(depth/2),
		learning_rate = [1e-3,1e-3],
		momentum = [0.9,0.9],
		num_epochs = int(5e4),
		minibatch_size = 64,
		log_file = 'log')
      
  #grid_search_layer_sizes()    
      
      