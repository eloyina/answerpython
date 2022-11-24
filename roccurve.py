import  matplotlib.pyplot  as  plt

#create roc curve
def roc_curve(y_true, y_pred):
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.show()

#calculate roc curve
#create data y_True 

y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
roc_curve(y_true, y_pred)





