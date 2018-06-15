import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve

from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

def cv2ShowWait(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

################################################################################
################################################################################
def pil2matRGB(pil_i, show=False):
    open_cv_image = np.array(pil_i) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    if show == True :
        cv2ShowWait(open_cv_image)
    return open_cv_image

################################################################################
################################################################################
def mat2pil(cv_i, show=False):
    cv2_im = cv2.cvtColor(cv_i,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_im)
    if show == True :
        image.show()
    return image

################################################################################
################################################################################
def plot_chart( imgs, x, y, fig_sze = 1, ax_vis = True, gray = False,
                title   = '',
                titles  = [],
                savefig = ''):
    
    #fig_sze -> (20,4)
    fig = plt.figure(fig_sze)
    fig.suptitle(title, fontsize=24, fontweight='bold')
    
    pos = 1
    col = 1
    row = 1

    if len(imgs) != len(titles):
        titles = ['' for  x in range(len(imgs)) ]

    for img in imgs:
        ax = plt.subplot(x, y, pos)
        if gray: plt.gray()
        plt.imshow(img)
        pos += 1
        ax.get_xaxis().set_visible(ax_vis)
        ax.get_yaxis().set_visible(ax_vis)
        ax.set_title(titles[pos-2])

    if len(savefig):
        print('Save figure in: ', savefig)
        plt.savefig(savefig)
    else:
        plt.show()
    

################################################################################
################################################################################
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

################################################################################
################################################################################
def psnr_(img1, img2, mask):
    temp = img1 * 0
    for i in range (img1.shape[0]):
        for j in range (img1.shape[1]):
            if mask[i,j] == 0:
                temp[i,j] = (img1[i,j]-img2[i,j]) ** 2

    mse = np.mean( temp )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

################################################################################
################################################################################
def plotImages(imgs, title, savefig =''):
    lines = np.floor( len(imgs) / 5 ) + 1
    plot_chart(imgs, lines, 5, title = title, savefig = savefig)

################################################################################
################################################################################
def showHistogram(data):
    plt.hist(data)
    plt.show()  

################################################################################
################################################################################
def ui_getERR(fpr, tpr):
    return brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

################################################################################
################################################################################
'''
it  receives obs which is a tuple with ([fpr], [tpr]) vectors,
must recieve labels for each vector
'''
def plotRocCurve(obs, labels):
    
    famous_colors = ['r', 'g', 'b', 'black','brown', 'slateblue', 'salmon', 'gold', 'gray', 'orange', 'violet' , 'darkgreen']

    pos = 0
   
    
    for fpr, tpr in obs: 
        roc_auc = np.trapz(x = fpr, y = tpr) *-1
        err     = ui_getERR(fpr, tpr)
        plt.plot(fpr, tpr, marker='.', color = famous_colors[pos], label = labels[pos] + '=%0.3f|' % roc_auc + 'ERR=%0.2f' % err ) 
        pos += 1

    plt.title('Receiver Operating Characteristic')
    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'darkgray', linestyle='--')
    plt.plot([0, 1], [1, 0],'darkgray', linestyle='--')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.show()
