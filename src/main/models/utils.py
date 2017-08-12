#*****************************
# Some utility functions
#
# Si Peng
# sipeng@adobe.com
# Jul. 14 2017
#*****************************


############ import some modules #####################
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import warnings
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.metrics import roc_auc_score, matthews_corrcoef, \
precision_recall_fscore_support, average_precision_score, \
roc_curve, auc, accuracy_score
from sklearn import preprocessing
######################################################



############ utility functions #######################
def load_data(data_path):
    """
    load training, validation and
    test data of hdf5
    """
    data_block      = h5py.File(data_path)
    np_feature_time = np.array(data_block['X'])
    np_churn        = np.array(data_block['y'])
    np_guid         = np.array(data_block['guid'])
    data_block.close()
    return np_feature_time, np_churn, np_guid


def load_data(data_path):
    """
    load training, validation and
    test data of hdf5
    """
    data_block      = h5py.File(data_path)
    seat_id         = np.array(data_block['seat_id'])
    contract_id     = np.array(data_block['contract_id'])
    X               = np.array(data_block['X'])
    y1_daily        = np.array(data_block['label_seat_daily'])
    y1_monthly      = np.array(data_block['label_seat_monthly'])
    y2_daily        = np.array(data_block['label_contract_daily'])
    y2_monthly      = np.array(data_block['label_contract_monthly'])
    seat_create     = np.array(data_block['seat_create'])
    seat_cancel     = np.array(data_block['seat_cancel'])
    contract_create = np.array(data_block['contract_create'])
    contract_cancel = np.array(data_block['contract_cancel'])
    data_block.close()
    return seat_id, contract_id, X, y1_daily, y1_monthly, y2_daily, y2_monthly, seat_create, seat_cancel, \
    contract_create, contract_cancel


def load_data_lstm1(data_path):
    """
    load and process data for layer 1 LSTM model
    """
    data_block  = h5py.File(data_path)
    X           = np.array(data_block['X'])
    y1_daily    = np.array(data_block['label_seat_daily'])
    seat_create = np.array(data_block['seat_create'])
    seat_cancel = np.array(data_block['seat_cancel'])
    data_block.close()
    timesteps   = X.shape[1]
    # transform data
    x = proc_pad(X, seat_create, seat_cancel)
    y = y1_daily.reshape((-1, timesteps, 1))
    return x, y


def load_pooling(data_path):
    """
    load the contract id for pooling
    Note: X_monthly in the returning values is masked with NAN, not -1.0,
    since we need to do scaling on X_pool
    """
    data_block  = h5py.File(data_path)
    contract_id = np.array(data_block['contract_id'])
    churn_label = np.array(data_block['label_contract_monthly'])
    data_block.close()
    return contract_id, churn_label


def load_LSTM1_monthly(path, weight = 50):
    """
    loads the data for LSTM1 monthly model from hdf5 files
    """
    ## read in data and process
    data_block  = h5py.File(path)
    X           = np.array(data_block['X'])
    y           = np.array(data_block['label_seat_monthly'])
    seat_create = np.array(data_block['seat_create'])
    data_block.close()
    
    X = np.where(X == -1, np.nan, X)
    y = y[:, 1:]  # don't predict in month 0
    
    ## aggregate X into 12 timesteps
    X_monthly = np.zeros((X.shape[0], 12, X.shape[2]))
    for i in range(0, 12):
        X_monthly[:, i, :] = np.nansum(X[:, 30*i:30*i+29, :], axis = 1)
    ## add age information into the feature, not good
    #age = np.repeat(seat_create, 12).reshape(-1, 12, 1)
    #X_monthly = np.concatenate((X_monthly, age), axis=2)
    ## use seat labels to mask X_monthly, since the sum of all nan's 
    ## are calculated as 0, instead of nan
    X_monthly = mask_tensor(X_monthly, y)
    
    ## create sample weight for y
    sw = np.empty((y.shape[0], y.shape[1], 1))
    sw[y == -1.] = 0
    sw[y == 0.]  = 1
    sw[y == 1.]  = weight
    sw = sw.reshape((sw.shape[0], -1))
    return X_monthly, y, sw


def load_LSTM2_monthly_baseline(path, weight = 50):
    """
    loads the data for LSTM2 monthly baseline model from hdf5 files
    Note: X_pool in the returning values is masked with NAN, not -1.0,
    since we need to do scaling on X_pool
    """
    ## read in data and process
    data_block = h5py.File(path)
    X          = np.array(data_block['X'])
    y          = np.array(data_block['label_contract_monthly'])
    y1_monthly = np.array(data_block['label_seat_monthly'])
    cid        = np.array(data_block['contract_id'])
    data_block.close()
    
    X = np.where(X == -1, np.nan, X)
    y = y[:, 1:]
    y1_monthly = y1_monthly[:, 1:]
    
    ## aggregate X into 12 timesteps
    X_monthly = np.zeros((X.shape[0], 12, X.shape[2]))
    for i in range(0, 12):
        X_monthly[:, i, :] = np.nansum(X[:, 30*i:30*i+29, :], axis = 1)
    
    ## use seat labels to mask X_monthly, since the sum of all nan's 
    ## are calculated as 0, instead of nan
    X_monthly = mask_tensor(X_monthly, y1_monthly)
    
    ## pool into contract level
    N = X_monthly.shape[2]  # dimension of individual features
    t = X_monthly.shape[1]  # number of timesteps
    cid_unique = np.unique(cid)
    # create empty np arrays
    n = len(cid_unique)
    X_pool = np.empty((n, t, N+1))
    labels = np.empty((n, t, 1))
    # the positions of each unique contract id
    cid_unique = np.unique(cid)
    contract_list = [np.argwhere(cid == i).T for i in cid_unique]
    for i in range(0, n):
        index_list = contract_list[i][0]
        X_pool[i, :, :-1]  = np.nansum(X_monthly[index_list], axis = 0)
        # number of seats, which are the non-NAN value counts
        cnt             = np.count_nonzero(~np.isnan(X_monthly[index_list, :, 0]), axis=0)
        cnt             = np.array(cnt, dtype=float)
        cnt[cnt == 0.]  = None
        X_pool[i, :, N] = cnt
        # correspoding labels
        labels[i] = y[index_list[0]]
    
    X_pool = mask_tensor(X_pool, labels)
    
    ## create sample weight for y
    sw = np.empty((n, t, 1))
    sw[labels == -1.] = 0
    sw[labels == 0.]  = 1
    sw[labels == 1.]  = weight
    sw = sw.reshape((sw.shape[0], -1))
    return X_pool, labels, sw


def load_lr_baseline(path, weight=50):
    """
    loads the data for logistic regression & mlp baseline model from hdf5 files
    Note: X_pool in the returning values is masked with NAN, not -1.0,
    since we need to do scaling on X_pool
    """
    ## read in data and process
    data_block = h5py.File(path)
    X          = np.array(data_block['X'])
    y          = np.array(data_block['label_contract_monthly'])
    y1_monthly = np.array(data_block['label_seat_monthly'])
    cid        = np.array(data_block['contract_id'])
    data_block.close()
    
    X = np.where(X == -1, np.nan, X)
    y = y[:, 1:]
    y1_monthly = y1_monthly[:, 1:]
    
    ## aggregate X into 12 timesteps
    X_monthly = np.zeros((X.shape[0], 12, X.shape[2]*2))
    for i in range(0, 12):
        # last 30days sum
        X_monthly[:, i, :X.shape[2]] = np.nansum(X[:, 30*i:30*i+29, :], axis = 1)
        # lifetime sum
        X_monthly[:, i, X.shape[2]:] = np.nansum(X[:, :30*i+29, :], axis = 1)
    
    ## use seat labels to mask X_monthly, since the sum of all nan's 
    ## are calculated as 0, instead of nan
    X_monthly = mask_tensor(X_monthly, y1_monthly)
    
    ## pool into contract level
    N = X_monthly.shape[2]  # dimension of individual features
    t = X_monthly.shape[1]  # number of timesteps
    cid_unique = np.unique(cid)
    # create empty np arrays
    n = len(cid_unique)
    X_pool = np.empty((n, t, N+1))
    labels = np.empty((n, t, 1))
    # the positions of each unique contract id
    cid_unique = np.unique(cid)
    contract_list = [np.argwhere(cid == i).T for i in cid_unique]
    for i in range(0, n):
        index_list = contract_list[i][0]
        X_pool[i, :, :-1]  = np.nansum(X_monthly[index_list], axis = 0)
        # number of seats, which are the non-NAN value counts
        cnt             = np.count_nonzero(~np.isnan(X_monthly[index_list, :, 0]), axis=0)
        cnt             = np.array(cnt, dtype=float)
        cnt[cnt == 0.]  = None
        X_pool[i, :, N] = cnt
        # correspoding labels
        labels[i] = y[index_list[0]]
    
    X_pool = mask_tensor(X_pool, labels)
    
    ## create sample weight for y
    sw = np.empty((n, t, 1))
    sw[labels == -1.] = 0
    sw[labels == 0.]  = 1
    sw[labels == 1.]  = weight
    sw = sw.reshape((sw.shape[0], -1))
    return X_pool, labels, sw


def npArray_to_hdf5(np_sid, np_cid, np_feature, np_seat_daily, np_seat_monthly, np_seat_create, np_seat_cancel, \
    np_contract_daily, np_contract_monthly, np_contract_create, np_contract_cancel, file_name, output_path):
    file_name = file_name.split('/')[-1] + '.hdf5'
    file_name = output_path + file_name
    h5f = h5py.File(file_name, 'w')
    h5f.create_dataset('seat_id', data = np_sid, compression="gzip", compression_opts=9)
    h5f.create_dataset('contract_id', data = np_cid, compression="gzip", compression_opts=9)
    h5f.create_dataset('X', data = np_feature, compression="gzip", compression_opts=9)
    h5f.create_dataset('label_seat_daily', data = np_seat_daily, compression="gzip", compression_opts=9)
    h5f.create_dataset('label_seat_monthly', data = np_seat_monthly, compression="gzip", compression_opts=9)
    h5f.create_dataset('seat_create', data = np_seat_create, compression="gzip", compression_opts=9)
    h5f.create_dataset('seat_cancel', data = np_seat_cancel, compression="gzip", compression_opts=9)
    h5f.create_dataset('label_contract_daily', data = np_contract_daily, compression="gzip", compression_opts=9)
    h5f.create_dataset('label_contract_monthly', data = np_contract_monthly, compression="gzip", compression_opts=9)
    h5f.create_dataset('contract_create', data = np_contract_create, compression="gzip", compression_opts=9)
    h5f.create_dataset('contract_cancel', data = np_contract_cancel, compression="gzip", compression_opts=9)
    h5f.close()


def split_data(path, files, cid):
    """
    use the cid list to filter the corresponding data
    """
    file            = files[0]
    data_block      = h5py.File(path + file)
    contract_id_tmp = np.array(data_block['contract_id'])
    flag            = np.in1d(contract_id_tmp, cid)
    # filter the data
    contract_id     = contract_id_tmp[flag]
    seat_id         = np.array(data_block['seat_id'])[flag]
    X               = np.array(data_block['X'])[flag]
    y1_daily        = np.array(data_block['label_seat_daily'])[flag]
    y1_monthly      = np.array(data_block['label_seat_monthly'])[flag]
    y2_daily        = np.array(data_block['label_contract_daily'])[flag]
    y2_monthly      = np.array(data_block['label_contract_monthly'])[flag]
    seat_create     = np.array(data_block['seat_create'])[flag]
    seat_cancel     = np.array(data_block['seat_cancel'])[flag]
    contract_create = np.array(data_block['contract_create'])[flag]
    contract_cancel = np.array(data_block['contract_cancel'])[flag]
    data_block.close()
    # go through the rest of the files
    for file in files[1:]:
        data_block      = h5py.File(path + file)
        contract_id_tmp = np.array(data_block['contract_id'])
        flag            = np.in1d(contract_id_tmp, cid)
        # filter the data
        contract_id     = np.concatenate((contract_id, contract_id_tmp[flag]), axis=0)
        seat_id         = np.concatenate((seat_id, np.array(data_block['seat_id'])[flag]), axis=0)
        X               = np.concatenate((X, np.array(data_block['X'])[flag]), axis=0)
        y1_daily        = np.concatenate((y1_daily, np.array(data_block['label_seat_daily'])[flag]), axis=0)
        y1_monthly      = np.concatenate((y1_monthly, np.array(data_block['label_seat_monthly'])[flag]), axis=0)
        y2_daily        = np.concatenate((y2_daily, np.array(data_block['label_contract_daily'])[flag]), axis=0)
        y2_monthly      = np.concatenate((y2_monthly, np.array(data_block['label_contract_monthly'])[flag]), axis=0)
        seat_create     = np.concatenate((seat_create, np.array(data_block['seat_create'])[flag]), axis=0)
        seat_cancel     = np.concatenate((seat_cancel, np.array(data_block['seat_cancel'])[flag]), axis=0)
        contract_create = np.concatenate((contract_create, np.array(data_block['contract_create'])[flag]), axis=0)
        contract_cancel = np.concatenate((contract_cancel, np.array(data_block['contract_cancel'])[flag]), axis=0)
        data_block.close()
    # process the data, correct the padding
    X          = proc_pad(X, seat_create, seat_cancel)
    y1_daily   = y1_daily.reshape((-1, y1_daily.shape[1], 1))
    y1_monthly = y1_monthly.reshape((-1, y1_monthly.shape[1], 1))
    y2_daily   = y2_daily.reshape((-1, y2_daily.shape[1], 1))
    y2_monthly = y2_monthly.reshape((-1, y2_monthly.shape[1], 1))
    return seat_id, contract_id, X, y1_daily, y1_monthly, seat_create, seat_cancel, \
    y2_daily, y2_monthly, contract_create, contract_cancel


def standardize_X(x, mean = [None], std = [None]):
    """
    standardize the feature tensor
    """
    x_norm = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
    if mean[0] == None:
        mean = np.nanmean(x_norm, axis=0)
        std  = np.nanstd(x_norm, axis=0)
        x_norm -= mean
        x_norm /= std
        x = x_norm.reshape(x.shape[0], x.shape[1], -1)
        return x, mean, std
    else:
        x_norm -= mean
        x_norm /= std
        x = x_norm.reshape(x.shape[0], x.shape[1], -1)
        return x

def scale_X(x, std = [None]):
    """
    standardize the feature tensor
    """
    x_norm = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
    if std[0] == None:
        std  = np.nanstd(x_norm, axis=0)
        x_norm /= std
        x = x_norm.reshape(x.shape[0], x.shape[1], -1)
        return x, std
    else:
        x_norm /= std
        x = x_norm.reshape(x.shape[0], x.shape[1], -1)
        return x


def replace_pad(X, pad_value):
    """
    replace the padded value with 0
    """
    X[X == pad_value] = 0
    return X


def proc_pad(X, create, cancel):
    """
    corrects the padding between create day and cancel day
    of the feature tensor
    """
    n = X.shape[0]
    t = X.shape[1]
    for i in range(0, n):
        if cancel[i] < t:
            X[i, create[i]:cancel[i]+1, :] = replace_pad(X[i, create[i]:cancel[i]+1, :], -1)
        else:
            X[i, create[i]:, :] = replace_pad(X[i, create[i]:, :], -1)
    return X


def squeeze(X):
    """
    squeeze the dimension of feature vectors from 14 to 2
    by taking the sum of the first 7 dimenstions, and the last 7
    """
    X_sq1 = X[:, :, :7].sum(axis=2).reshape((X.shape[0], -1, 1))
    X_sq2 = X[:, :, 7:].sum(axis=2).reshape((X.shape[0], -1, 1))
    X_sq  = np.concatenate((X_sq1, X_sq2), axis=2)
    return X_sq


def mask_tensor(X, y):
    """
    masks the input tensor by NA
    with the mask pattern in y
    """
    mask = (y != -1).astype(float)
    mask[mask == 0] = None  # set the masked position to have None value
    X *= mask
    return X


def expand_y(y, timesteps):
    """
    create the sample weight matrix for the output
    """
    n      = y.shape[0] # number of samples
    t      = y.shape[1]
    sw     = np.zeros((n, timesteps))
    labels = np.zeros((n, timesteps, 1))
    for i in range(0, t):
        sw[:, 30*i]        = (1, ) * (y[:, i] != -1.0)
        labels[:, 30*i, 0] = y[:, i]
    return labels, sw


def pool(X, cid, churn_label):
    """
    pool the individual features into group features,
    also output the churn labels
    """
    N = X.shape[2]  # dimension of individual features
    t = X.shape[1]  # number of timesteps
    cid_unique = np.unique(cid)
    # create empty np arrays
    n = len(cid_unique)
    X_pool = np.empty((n, t, 3*N+1))
    labels = np.empty((n, churn_label.shape[1]))
    # the positions of each unique contract id
    contract_list = [np.argwhere(cid == i).T for i in cid_unique]
    for i in range(0, len(contract_list)):
        index_list = contract_list[i][0]
        X_pool[i, :, 0:N]     = np.mean(~np.isnan(X[index_list]), axis = 0) # mean
        X_pool[i, :, N:2*N]   = np.max(~np.isnan(X[index_list]), axis = 0) # maximum
        X_pool[i, :, 2*N:3*N] = np.min(~np.isnan(X[index_list]), axis = 0) # minimum
        # number of seats, which are the non-NAN value counts
        cnt               = np.count_nonzero(~np.isnan(X[index_list, :, 0]), axis=0)
        cnt               = np.array(cnt, dtype=float)
        cnt[cnt == 0.]    = None
        X_pool[i, :, 3*N] = cnt
        # correspoding labels
        labels[i] = churn_label[index_list[0]][:, 0]
    labels, weights = expand_y(labels, t) # both becomes 3d arrays
    X_pool[np.isnan(X_pool)] = -1.0
    return X_pool, labels, weights


def pool_monthly(X, cid, churn_label, weight = 50):
    """
    pool the individual features into group features,
    also output the churn labels
    """
    N = X.shape[2]  # dimension of individual features
    t = X.shape[1]  # number of timesteps
    cid_unique = np.unique(cid)
    # create empty np arrays
    n = len(cid_unique)
    X_pool = np.empty((n, t, 3*N+1))
    labels = np.empty((n, churn_label.shape[1], 1))
    # the positions of each unique contract id
    contract_list = [np.argwhere(cid == i).T for i in cid_unique]
    for i in range(0, len(contract_list)):
        index_list = contract_list[i][0]
        # number of seats, which are the non-NAN value counts
        cnt               = np.count_nonzero(~np.isnan(X[index_list, :, 0]), axis=0)
        cnt               = np.array(cnt, dtype=float)
        cnt[cnt == 0.]    = None
        X_pool[i, :, 3*N] = cnt
        ## in the following, the mean, max and min of an array of all NAN's
        ## will be just NAN, which needs attention later on
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            X_pool[i, :, 0:N]     = np.nanmean(X[index_list], axis = 0) # mean
            X_pool[i, :, N:2*N]   = np.nanmax(X[index_list], axis = 0) # maximum
            X_pool[i, :, 2*N:3*N] = np.nanmin(X[index_list], axis = 0) # minimum
        ## replace all NAN's with 0, but the true NAN's are also replaced
        X_pool[np.isnan(X_pool)] = 0.0
        # correspoding labels
        labels[i] = churn_label[index_list[0]]
    # fix the masking, replace 0's by NAN for the masked timesteps
    X_pool = mask_tensor(X_pool, labels)
    ## create sample weight for y
    sw = np.empty((n, t, 1))
    sw[labels == -1.] = 0
    sw[labels == 0.]  = 1
    sw[labels == 1.]  = weight
    sw = sw.reshape((sw.shape[0], -1))
    return X_pool, labels, sw


def save_history(file_name, history_var):
    """
    save the training loss and validation loss from history in the file
    """
    file_name = file_name + '.hdf5'
    h5f = h5py.File(file_name, 'w')
    h5f.create_dataset('train_loss', data = history_var.history['loss'], compression="gzip", compression_opts=9)
    h5f.create_dataset('valid_loss', data = history_var.history['val_loss'], compression="gzip", compression_opts=9)
    h5f.close()


def load_history(file_name):
    """
    load the training and validation losses in the file
    """
    hist       = h5py.File(file_name, 'r')
    train_loss = hist['train_loss']
    valid_loss = hist['valid_loss']
    return train_loss, valid_loss


def model_AUC(y_test, y_pred, weight):
    """
    a wrapper for AUC computation
    """
    y_test      = y_test.reshape((-1))
    y_pred      = y_pred.reshape((-1))
    weight      = weight.reshape((-1))
    fpr, tpr, _ = roc_curve(y_test*weight, y_pred*weight, sample_weight=weight)
    aur_ROC     = auc(fpr, tpr)
    #auc_PR  = average_precision_score(y_test, y_pred, average="micro", sample_weight=weight)
    return auc_ROC, fpr, tpr


def plot_loss_curve(path, file_name, train_loss, valid_loss, k = 0):
    """
    plot the training and validation loss curves
    """
    epoch = range(1,len(train_loss)+1)
    plt.plot(epoch[k:], train_loss[k:])
    plt.plot(epoch[k:], valid_loss[k:])
    plt.legend(['train_loss', 'valid_loss'], loc = 'upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(file_name)
    plt.savefig(path + file_name + '.png')
    plt.close()


def plot_roc(path, file_name, fpr, tpr):
    auc_ROC = np.around(auc(fpr, tpr), decimals=3)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.text(0.8, 0.05, "AUC="+str(auc_ROC))
    plt.title(file_name)
    plt.savefig(path + file_name + '.png')
    plt.close()


def plot_pr(path, file_name, rec, pre):
    auc_PR = np.around(auc(rec, pre), decimals=3)
    plt.plot(rec, pre)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.text(0.8, 0.9, "AUC="+str(auc_PR))
    plt.title(file_name)
    plt.savefig(path + file_name + '.png')
    plt.close()


def plot_model_architecture(model, figure_name = 'model.png'):
    """
    plot the model structure
    """
    plot_model(model, to_file=figure_name, show_shapes = True, show_layer_names = True)

