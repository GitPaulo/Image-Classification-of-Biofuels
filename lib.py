# Commonly used functions in the project
import os
import numpy as np
import _pickle as cPickle

from time import gmtime, strftime
from matplotlib import pyplot as plt
from matplotlib import pyplot as pp
from matplotlib import image as mpimg
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

def stringify(something):
    if type(something) == list:
        return [stringify(x) for x in something]
    elif type(something) == tuple:
        return tuple(stringify(list(something)))
    else:
        return str(something)
    
def log(*msg):
    msg = stringify(msg)
    print(strftime("[%H:%M:%S]", gmtime()), " ".join(msg))
    
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return ax

def shuffleRawDataset(data, labels):
    result = np.arange(data.shape[0])
    np.random.shuffle(result)
    return data[result], labels[result]

def shuffleJoinRawDatasets(data1, labels1, data2, labels2):
    data1, labels1 = shuffleRawDataset(data1, labels1)
    data2, labels2 = shuffleRawDataset(data2, labels2)
    return np.concatenate((data1, data2)), np.concatenate((labels1, labels2))

def saveTrainedModel(model, folderName="NewModel", fileName="model"):
    folderPath = "app\\trained_models\\" + folderName
    
    # Check if folder exists, otherwise creat
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    
    # Dump model
    with open(folderPath + "\\" + fileName + '.pkl', 'wb') as fid:
        cPickle.dump(model, fid)

# Tool to display data set and its labels
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
	if type(ims[0]) is np.ndarray:
		ims = np.array(ims).astype(np.uint8)
		if (ims.shape[-1] != 3):
			ims = ims.transpose((0,2,3,1))
	f = pp.figure(figsize=figsize)
	cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
	for i in range(len(ims)):
		sp = f.add_subplot(rows, cols, i+1)
		sp.axis('Off')
		if titles is not None:
			sp.set_title(titles[i], fontsize=16)
		pp.imshow(ims[i], interpolation=None if interp else 'none')
        
log("Library functions loaded.")