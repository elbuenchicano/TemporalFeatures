import os

from utils import *

  
################################################################################
################################################################################
################################ Main controler ################################
'''
Maincontroler class
JSON config.
must have 
'funtion_id' corresponding to dict with specific parameters
'general' with general imformation for whole framework
'[funtion_id]' dict with specific information 
    it must have: 
    'file_list'
'''
class MainControler:
    def __init__(self, main_functions, json_file='conf.json'):
        conf            = u_getPath(json_file)
        self.confs_     = json.load(open(conf))
        self.func_dict_ = main_functions

    #...........................................................................
    '''
    This f collects information from json 
    returns (a,b,c)
    a = parameters dict
    b = [file(s)] to compute
    '''
    def getInfo(self): 
        individual  = self.confs_[self.confs_['function_id']]
        file_list   = u_loadFileManager( individual['file_list'])
        return  {**self.confs_['general'], **individual}, file_list

    #...........................................................................
    '''
    This funtions post processing the whole data if previous process return 
    something
    '''
    def postProcess(self, collected): pass

    #...........................................................................
    '''
    Main funtion that joins the abstract process
    '''
    def run(self):
        if 'run_flag' in self.confs_:
            if not self.confs_['run_flag']:
                #self.func_dict_[self.confs_['function_id']]
                #( 
                #  {**self.confs_['general'], 
                #   **self.confs_[self.confs_['function_id']]
                #  }
                #)
                self.func_dict_[self.confs_['function_id']]('', {**self.confs_['general'], **self.confs_[self.confs_['function_id']]})
            else:
                self.conventional_pipeline()
        else:
           self.conventional_pipeline()

    #...........................................................................
    #...........................................................................
    def conventional_pipeline(self):
        params, fvector   = self.getInfo()
        collected_out   = []
        final           = len(fvector)
        for i in range(final):
            out = self.func_dict_[self.confs_['function_id']](fvector[i], params)
            u_progress(i, final, 'I-', i)
   
            if out is not None:
                collected_out.append(out)

        if len(collected_out) > 0:
            self.postProcess(collected_out)

################################################################################
################################################################################