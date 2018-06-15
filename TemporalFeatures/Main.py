from utils      import *
from interface  import MainControler
from ReAnomalousTrajectory import *
   
################################################################################
################################################################################

################################################################################
################################################################################
class Controler(MainControler):
    def __init__(self, func_dict, json_file = 'conf.json'):
        return super().__init__(func_dict, json_file)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def postProcess(self, collected):
        pass

################################################################################
################################################################################
############################### MAIN ###########################################
if __name__ == '__main__':
    main_functions = {
                        'train'             : re_an_train,
                        'feat_extraction'   : featExtraction,
                        'clustering'        : clustering,
                        'visualize'         : visualizeClusters,
                        "knn"               : knn,
                        "validation"        : validatation,
                        "validation2"       : validation2,
                        "plot_roc"          : plotRoc,
                        "joinFeats"         : joinFeatures,
                        "tsne"              : tsne,
                        'join_metrics'      : joinMetrics
                     } 
    control =  Controler(main_functions, 'TemporalFeatures.json')
    control.run()