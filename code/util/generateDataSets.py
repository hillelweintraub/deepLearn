# python functions to convert my hyperspectral image datasets from
# .mat files to pickle files, and also to do som reformatting on
# the side

def get_mat_data(mat_file):
  '''
  Opens a .mat file and returns the array data stored within
  This assumes that only one variable is stored in the file 
  '''

  import scipy.io as sio
  import numpy

  mat_dict = sio.loadmat(mat_file)
  for value in mat_dict.itervalues():
    if(type(value)==numpy.ndarray):
      data = value
      break
  return data

def pickle_dataset(data,data_gt,file_name):
  """
  Saves the data set stored in the variables
  data and data_gt to a pickle file named file_name
  """
  
  import cPickle
  save_file = open(file_name,'wb')
  cPickle.dump(data,save_file,-1)
  cPickle.dump(data_gt,save_file,-1)
  save_file.close()

def load_pickled_dataset(file_name):
  """
  Loads a dataset pickled using the pickle_dataset function
  returns: data,data_gt
  """

  import cPickle
  save_file = open(file_name)
  data = cPickle.load(save_file)
  data_gt = cPickle.load(save_file)
  save_file.close()
  return data,data_gt 

def process_and_pickle_matfiles(data_file,gt_file,save_file):
  """
  This function loads a hyperspectral dataset from two .mat
  files ( a data file and a ground-truth file) and preprocesses
  it by reshaping the data, normalizing it, and getting rid of
  unlabeled data. It then saves the data to a pickle file
  named save_file
  
  Input:
    data_file - A .mat file with the hyperspectral data cube
    gt_file   - A .mat file with the class labels
    save_file - Name of the pickle file where the processed dat 
                will be saved
  Output: 
    A pickle file with two variable:
      data:
        -shape: (num_pixels,num_spectral_bands)
        -Data normalized to range [0,1]

      data_gt:
        -shape: (num_pixels,)
        -class labels
  """

  #load the data and the ground truth class labels 
  data = get_mat_data(data_file)
  data_gt = get_mat_data(gt_file)
 
  m,n,k = data.shape
  assert (m,n)==data_gt.shape, \
         "Errorr: shape mismatch between data and groundtruth"

  #reshape the data
  data = data.reshape((m*n,k),order='C')
  data_gt = data_gt.reshape(m*n,order='C')
  #get rid of class 0 (unlabeled pixels)
  data = data[data_gt!=0,:]
  data_gt = data_gt[data_gt!=0]
  #normalize data to range [0,1]
  data = data/float( data.max() )
  #save the data
  pickle_dataset(data,data_gt,save_file)
  

def split_to_train_val_test(file_name,split_proportions=[6,2,2],random_split=False,remove_outliers=True):
  """
    Loads data, and data_gt variables from a dataset file and splits the data into 
    training, validation, and test data using a given split proportion.
    
    Inputs:
      file_name - the name of a pickle file created using pickle_dataset that contains 
                  the training examples and ground truth labels
      split_proportions - an array containig the relative split proportion values for
                          the trainig, validation,and testing data respectively
      random_split - a boolean flag for whether to use a random shuffle before creating 
                     the data sets. This allows for running multiple experiments varying
                     the exact examples contained in each set
    Outpus:
      train_set, valid_set, test_set format: tuple(input, target)
      input is an numpy.ndarray of 2 dimensions (a matrix)
      with row's correspond to an example. target is a
      numpy.ndarray of 1 dimensions (vector)) that have the same length as
      the number of rows in the input. It should give the target
      to the example with the same index in the input.                          
  """
  
  import random
  import numpy
  
  assert len(split_proportions)==3, ("Error: expecting a split proportion value for "
                                      "training validation and test sets")
  data,data_gt = load_pickled_dataset(file_name)
  #remove outliers
  if remove_outliers:
    flen=len(file_name)
    assert file_name[flen-7:flen]=='KSC.pkl', 'Recheck outliers for new dataset'
    maxes = numpy.max(data,axis=1)
    data = data[maxes<.9,:]
    data_gt = data_gt[maxes<.9]
    #renormalize data
    data = data/float( data.max() )
  
  #we need the class labels to start from 0 so shift them if they  don't
  if data_gt.min() == 1: data_gt -= 1
  p = range(len(data))  
  #we need to shuffle the data to get a uniform sampling of class examples in
  #each of the train,validation ,and test sets.
  #But, we'll seed the rng to get the same random shuffling every time unless
  #random_split is set to True
  if not random_split: random.seed(12345)
  random.shuffle(p)
  split_idxs = len(data) * numpy.cumsum(split_proportions)/numpy.sum(split_proportions)
  
  train_idxs = p[0:split_idxs[0]]
  train_input = data[train_idxs,:]
  train_target = data_gt[train_idxs]
  train_set = (train_input,train_target)
 
  valid_idxs = p[split_idxs[0]:split_idxs[1]]
  valid_input = data[valid_idxs,:]
  valid_target = data_gt[valid_idxs]
  valid_set = (valid_input,valid_target)
 
  test_idxs = p[split_idxs[1]:split_idxs[2]]
  test_input = data[test_idxs,:]
  test_target = data_gt[test_idxs]
  test_set = (test_input,test_target)
 
  return  train_set, valid_set, test_set



