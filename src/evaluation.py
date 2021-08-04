import torch

def confusion_matrix(prediction, label, one_hot_encoded=True):
    '''
    returns the confusion matrix of the given prediction.
    axis:
    0 = label
    1 = prediction
    '''
    if one_hot_encoded:
        _, classes, *_ = label.shape
    else:
        classes = int(label.max())

    prediction = torch.argmax(prediction, dim=1)
    prediction = prediction.flatten()
    if one_hot_encoded:
        label = torch.argmax(label, dim=1)
    label = label.flatten()

    confusion_matrix = torch.zeros((classes, classes))

    for i in range(classes):
        class_specific_prediction = prediction[label==i]
        for j in range(classes):
            confusion_matrix[i,j] = torch.sum(class_specific_prediction == j)

    return confusion_matrix

def overall_accuracy(confusion_matrix, ignore_index_0=False):
    '''
    see https://gis.humboldt.edu/OLM/Courses/GSP_216_Online/lesson6-2/metrics.html#:~:text=The%20User's%20Accuracy%20is%20the,user%2C%20not%20the%20map%20maker.&text=The%20User's%20Accuracy%20is%20calculating,it%20by%20the%20row%20total.
    '''
    if ignore_index_0:
        confusion_matrix = confusion_matrix[1:, 1:]
    overall_accuracy = 0
    for i in range(len(confusion_matrix)):
        overall_accuracy += confusion_matrix[i,i]
    overall_accuracy /= torch.sum(confusion_matrix)
    return overall_accuracy
    
def mean_users_accuracy(confusion_matrix, ignore_index_0=False):
    '''
    see https://gis.humboldt.edu/OLM/Courses/GSP_216_Online/lesson6-2/metrics.html#:~:text=The%20User's%20Accuracy%20is%20the,user%2C%20not%20the%20map%20maker.&text=The%20User's%20Accuracy%20is%20calculating,it%20by%20the%20row%20total.
    '''
    if ignore_index_0:
        confusion_matrix = confusion_matrix[1:, 1:]
    
    valid_classes = 0
    mu_accuracy = 0
    for i in range(len(confusion_matrix)):
        divider = torch.sum(confusion_matrix[:,i])
        if divider == 0:
            continue
        mu_accuracy += confusion_matrix[i,i] / divider
        valid_classes += 1
    if valid_classes == 0:
        return 0
    mu_accuracy /= valid_classes
    return mu_accuracy

def mean_producers_accuracy(confusion_matrix, ignore_index_0=False):
    '''
    see https://gis.humboldt.edu/OLM/Courses/GSP_216_Online/lesson6-2/metrics.html#:~:text=The%20User's%20Accuracy%20is%20the,user%2C%20not%20the%20map%20maker.&text=The%20User's%20Accuracy%20is%20calculating,it%20by%20the%20row%20total.
    '''
    if ignore_index_0:
        confusion_matrix = confusion_matrix[1:, 1:]
    mp_accuracy = 0
    for i in range(len(confusion_matrix)):
        mp_accuracy += confusion_matrix[i,i] /  torch.sum(confusion_matrix[i])
    mp_accuracy /= len(confusion_matrix)
    return mp_accuracy

def kappa_coefficent(confusion_matrix, ignore_index_0=False):
    '''
    see https://www.medistat.de/glossar/uebereinstimmung/cohens-kappa-koeffizient#:~:text=Der%20Kappa%2DKoeffizient%20nach%20Cohen,einfache%20Bewertung%20zweier%20verschiedener%20Beurteiler.
    '''
    if ignore_index_0:
        confusion_matrix = confusion_matrix[1:, 1:]

    p0 = 0
    pe = 0
    for i in range(len(confusion_matrix)):
        p0 += confusion_matrix[i,i]
        pe += torch.sum(confusion_matrix[:,i]) * torch.sum(confusion_matrix[i])
    N = len(confusion_matrix) 
    p0 /= N
    pe /= N*N
    return (p0 - pe)/(1 - pe)