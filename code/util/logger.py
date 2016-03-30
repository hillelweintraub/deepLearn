
# A Logger class for logging experiment runs 

class Logger:
  """
  A simple class for logging stuff
  """
  
  def __init__(self,logfile):
    """
    Constructor. Opens a file object for writing
    """
    self.myLog = open(logfile,'w')
    
  def log(self, log_string, print_stdout=True):
    """
    write log_string to the log. If print_stdout, also print log_string to stdout
    """
    self.myLog.write(log_string+'\n')
    self.flush()
    if print_stdout: print log_string
    
  def flush(self):
    """
    Flush the IO buffer, to force a write to disc
    """
    self.myLog.flush()
 
  def add_newline(self,print_stdout=True):
    """
    Add a blank line to the log, and also to stdout if print_stdout is True
    """
    self.log("",print_stdout)
 
  def close(self):
    """
    Close the file object
    """
    self.myLog.close()
    
 