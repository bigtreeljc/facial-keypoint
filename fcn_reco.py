import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, time, os, warnings, cv2
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data as data
warnings.filterwarnings("ignore")

FTRAIN = "data/training.csv"
FTEST  = "data/test.csv"
FIdLookup = 'data/IdLookupTable.csv'

def gaussian_k(x0,y0,sigma, width, height):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float) ## (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def generate_hm(height, width ,landmarks,s=3):
    """ Generate a full Heap Map for every landmarks in an array
    Args:
        height    : The height of Heat Map (the height of target output)
        width     : The width  of Heat Map (the width of target output)
        joints    : [(x1,y1),(x2,y2)...] containing landmarks
        maxlenght : Lenght of the Bounding Box
    """
    Nlandmarks = len(landmarks)
    hm = np.zeros((height, width, Nlandmarks), dtype=np.float32)
    for i in range(Nlandmarks):
        if not np.array_equal(landmarks[i], [-1,-1]):
            hm[:,:,i] = gaussian_k(landmarks[i][0],
                landmarks[i][1], s, height, width)
        else:
            hm[:,:,i] = np.zeros((height, width))
    return hm

def get_y_as_heatmap(df,height,width, sigma):
    columns_lmxy = df.columns[:-1] ## the last column contains Image
    # print(columns_lmxy)
    columns_lm = []
    for c in columns_lmxy:
        c = c[:-2]
        if c not in columns_lm:
            columns_lm.extend([c])

    y_train = []
    for i in range(df.shape[0]):
        landmarks = []
        for colnm in columns_lm:
            x = df[colnm + "_x"].iloc[i]
            y = df[colnm + "_y"].iloc[i]
            if np.isnan(x) or np.isnan(y):
                x, y = -1, -1
            landmarks.append([x,y])

        y_train.append(generate_hm(height, width, landmarks, sigma))
    y_train = np.array(y_train)

    return(y_train,df[columns_lmxy],columns_lmxy)


def load(test=False, width=96,height=96,sigma=5):
    from sklearn.utils import shuffle

    fname = FTEST if test else FTRAIN
    df = pd.read_csv(os.path.expanduser(fname))

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    myprint = df.count()
    myprint = myprint.reset_index()
    print(myprint)
    ## row with at least one NA columns are removed!
    ## df = df.dropna()
    df = df.fillna(-1)

    X = np.vstack(df['Image'].values) / 255.  # changes valeus between 0 and 1
    X = X.astype(np.float32)

    if not test:  # labels only exists for the training data
        y, y0, nm_landmark = get_y_as_heatmap(df,height,width, sigma)
        X, y, y0 = shuffle(X, y, y0, random_state=42)  # shuffle data
        y = y.astype(np.float32)
    else:
        y, y0, nm_landmark = None, None, None

    return X, y, y0, nm_landmark

def load2d(test=False, width=96, height=96, sigma=5):
    re   = load(test,width,height,sigma)
    X    = re[0].reshape(-1,width,height,1)
    y, y0, nm_landmarks = re[1:]

    return X, y, y0, nm_landmarks

###############################################################################3
# 
# data loader of the kaggle data
#
###############################################################################

sigma = 5
X_train, y_train, y_train0, nm_landmarks = load2d(test=False, sigma=sigma)
X_test,  y_test, _, _ = load2d(test=True,sigma=sigma)

print(X_train.shape,y_train.shape, y_train0.shape)
print(X_test.shape,y_test)


###############################################################################3
# 
# data plotting of some heat maps
#
###############################################################################
Nplot = y_train.shape[3]+1
n_plot = 1
for i in range(n_plot):
    fig = plt.figure(figsize=(20,6))
    ax = fig.add_subplot(2,Nplot/2,1)
    ax.imshow(X_train[i,:,:,0],cmap="gray")
    ax.set_title("input")
    for j, lab in enumerate(nm_landmarks[::2]):
        ax = fig.add_subplot(2,Nplot/2,j+2)
        ax.imshow(y_train[i,:,:,j],cmap="gray")
        ax.set_title(str(j) +"\n" + lab[:-2] )
    plt.show()

###############################################################################3
# 
# data augmentation
#
###############################################################################
from skimage import transform
from skimage.transform import SimilarityTransform, AffineTransform
import random

landmark_order = {"orig" : [0,1,2,3,4,5,6,7,8,9,11,12],
                  "new"  : [1,0,4,5,2,3,8,9,6,7,12,11]}

def transform_img(data,
                  loc_w_batch=2,
                  max_rotation=0.01,
                  max_shift=2,
                  max_shear=0,
                  max_scale=0.01,mode="edge"):
    '''
    data : list of numpy arrays containing a single image
    e.g., data = [X, y, w] or data = [X, y]
    X.shape = (height, width, NfeatX)
    y.shape = (height, width, Nfeaty)
    w.shape = (height, width, Nfeatw)
    NfeatX, Nfeaty and Nfeatw can be different

    affine transformation for a single image

    loc_w_batch : the location of the weights in the fourth dimention
    [,,,loc_w_batch]
    '''
    scale = (np.random.uniform(1-max_scale, 1 + max_scale),
             np.random.uniform(1-max_scale, 1 + max_scale))
    rotation_tmp = np.random.uniform(-1*max_rotation, max_rotation)
    translation = (np.random.uniform(-1*max_shift, max_shift),
                   np.random.uniform(-1*max_shift, max_shift))
    shear = np.random.uniform(-1*max_shear, max_shear)
    tform = AffineTransform(
            scale=scale,#,
            ## Convert angles from degrees to radians.
            rotation=np.deg2rad(rotation_tmp),
            translation=translation,
            shear=np.deg2rad(shear)
        )

    for idata, d in enumerate(data):
        if idata != loc_w_batch:
            ## We do NOT need to do affine transformation for weights
            ## as weights are fixed for each (image,landmark) combination
            data[idata] = transform.warp(d, tform,mode=mode)
    return data


def horizontal_flip(data, lm, loc_y_batch=1, loc_w_batch=2):
    '''
    flip the image with 50% chance

    lm is a dictionary containing "orig" and "new" key
    This must indicate the potitions of heatmaps that need to be flipped
    landmark_order = {"orig" : [0,1,2,3,4,5,6,7,8,9,11,12],
                      "new"  : [1,0,4,5,2,3,8,9,6,7,12,11]}

    data = [X, y, w]
    w is optional and if it is in the code, the position needs to be specified
    with loc_w_batch

    X.shape (height,width,n_channel)
    y.shape (height,width,n_landmarks)
    w.shape (height,width,n_landmarks)
    '''
    lo, ln = np.array(lm["orig"]), np.array(lm["new"])

    assert len(lo) == len(ln)
    if np.random.choice([0,1]) == 1:
        return(data)

    for i, d in enumerate(data):
        d = d[:, ::-1,:]
        data[i] = d

    data[loc_y_batch] = swap_index_for_horizontal_flip(
        data[loc_y_batch], lo, ln)

    # when horizontal flip happens to image, we need to heatmap (y) and weights y and w
    # do this if loc_w_batch is within data length
    if loc_w_batch < len(data):
        data[loc_w_batch] = swap_index_for_horizontal_flip(
            data[loc_w_batch], lo, ln)
    return(data)

def swap_index_for_horizontal_flip(y_batch, lo, ln):
    '''
    lm = {"orig" : [0,1,2,3,4,5,6,7,8,9,11,12],
          "new"  : [1,0,4,5,2,3,8,9,6,7,12,11]}
    lo, ln = np.array(lm["orig"]), np.array(lm["new"])
    '''
    y_orig = y_batch[:,:, lo]
    y_batch[:,:, lo] = y_batch[:,:, ln]
    y_batch[:,:, ln] = y_orig
    return(y_batch)

def transform_imgs(data, lm,
                   loc_y_batch = 1,
                   loc_w_batch = 2):
    '''
    data : list of numpy arrays containing a single image
    e.g., data = [X, y, w] or data = [X, y]
    X.shape = (height, width, NfeatX)
    y.shape = (height, width, Nfeaty)
    w.shape = (height, width, Nfeatw)
    NfeatX, Nfeaty and Nfeatw can be different

    affine transformation for a single image
    '''
    Nrow  = data[0].shape[0]
    Ndata = len(data)
    data_transform = [[] for i in range(Ndata)]
    for irow in range(Nrow):
        data_row = []
        for idata in range(Ndata):
            data_row.append(data[idata][irow])
        ## affine transformation
        data_row_transform = transform_img(data_row,
                                          loc_w_batch)
        ## horizontal flip
        data_row_transform = horizontal_flip(data_row_transform,
                                             lm,
                                             loc_y_batch,
                                             loc_w_batch)

        for idata in range(Ndata):
            data_transform[idata].append(data_row_transform[idata])

    for idata in range(Ndata):
        data_transform[idata] = np.array(data_transform[idata])

    return(data_transform)

Nhm = 10
count = 1
Nplot = 5

iexample = 139
plt.imshow(X_train[iexample,:,:,0],cmap="gray")
plt.title("original")
plt.axis("off")
plt.show()
Nplot = 5
fig = plt.figure(figsize=[Nhm*2.5,2*Nplot])


landmark_order = {"orig" : [0,1,2,3,4,5,6,7,8,9,11,12],
                  "new"  : [1,0,4,5,2,3,8,9,6,7,12,11]}

for _ in range(Nplot):
    x_batch, y_batch = transform_imgs([X_train[[iexample]],
        y_train[[iexample]]], landmark_order)
    ax = fig.add_subplot(Nplot, Nhm+1, count)
    ax.imshow(x_batch[0, :, :, 0], cmap='gray')
    ax.axis("off")
    count += 1
    for ifeat in range(Nhm):
        ax = fig.add_subplot(Nplot, Nhm+1, count)
        ax.imshow(y_batch[0, :, :, ifeat], cmap="gray")
        ax.axis("off")
        if count < Nhm + 2:
            ax.set_title(nm_landmarks[ifeat*2][:-2])
        count += 1
plt.show()

###############################################################################3
# 
# data plotting of some heat maps
#
###############################################################################
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import torch.nn.functional as F

prop_train = 0.9
Ntrain = int(X_train.shape[0]*prop_train)
X_tra, y_tra, X_val, y_val = X_train[:Ntrain],y_train[:Ntrain],X_train[Ntrain:],y_train[Ntrain:]
'''
just for testing
'''
# X_tra = X_tra[:100]
# y_tra = y_tra[:100]
del X_train, y_train
input_height, input_width = 96, 96
## output shape is the same as input
output_height, output_width = input_height, input_width
n = 32*5
nClasses = 15
nfmp_block1 = 64
nfmp_block2 = 128
nb_epochs = 15
batch_size = 4
const = 10
history = {"loss":[], "val_loss":[]}

IMAGE_ORDERING = "channels_last"
img_input = Variable(torch.FloatTensor(batch_size, input_height, input_width, 1)).cuda()
img_label = Variable(torch.FloatTensor(batch_size, input_height, input_width, 15)).cuda()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, nfmp_block1, 3, padding=1)
        self.act1 = F.relu
        self.conv2 = nn.Conv2d(nfmp_block1, nfmp_block1, 3, padding=1)
        self.act2 = F.relu
        self.block1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(nfmp_block1, nfmp_block2, 3, padding=1)
        self.act3 = F.relu
        self.conv4 = nn.Conv2d(nfmp_block2, nfmp_block2, 3, padding=1)
        self.act4 = F.relu
        self.block2 = nn.MaxPool2d(2)

        ## bottoleneck
        # self.bl_conv1 = nn.Conv2d(nfmp_block2, n, (input_height//4, input_width//4),
        #     padding='same') 
        '''
        doing padding = same if odd sized kernel
        '''
        cur_h = input_height//4
        cur_w = input_width//4
        pad_h = cur_h//2
        pad_w = cur_w//2
        if cur_h % 2 == 0:
            pad_h1 = pad_h
            pad_h2 = pad_h - 1
        else:
            pad_h1 = pad_h2 = pad_h
        if cur_w % 2 == 0:
            pad_w1 = pad_w
            pad_w2 = pad_w - 1
        else:
            pad_w1 = pad_w2 = pad_w
        self.bl_pad = nn.ZeroPad2d((pad_h1, pad_h2, pad_w1, pad_w2))
        self.bl_conv1 = nn.Conv2d(nfmp_block2, n, (cur_h, cur_w), padding=0)
        self.bl_conv2 = nn.Conv2d(n, n, 1, padding=0)
        self.relu = F.relu
        '''
        upsample
        '''
        self.upsample = nn.ConvTranspose2d(n, nClasses, (4, 4), stride=(4, 4))
        
    def forward(self, img):
        x = self.conv1(img)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.block1(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.block2(x)
        x = self.relu(self.bl_conv1(self.bl_pad(x)))
        x = self.relu(self.bl_conv2(x))
        x = self.upsample(x)
        x = x.view(-1, output_width * output_height * nClasses, 1)
        return x


def find_weight(y_val):
    '''
    :::input:::

    y_val : np.array of shape (N_image, height, width, N_landmark)

    :::output:::

    weights :
        np.array of shape (N_image, height, width, N_landmark)
        weights[i_image, :, :, i_landmark] = 1
                        if the (x,y) coordinate of the landmark for this image is recorded.
        else  weights[i_image, :, :, i_landmark] = 0

    '''
    weight = np.zeros_like(y_val)
    count0, count1 = 0, 0
    for irow in range(y_val.shape[0]):
        for ifeat in range(y_val.shape[-1]):
            if np.all(y_val[irow,:,:,ifeat] == 0):
                value = 0
                count0 += 1
            else:
                value = 1
                count1 += 1
            weight[irow,:,:,ifeat] = value
    print("N landmarks={:5.0f}, N missing landmarks={:5.0f}, weight.shape={}".format(
        count0, count1,weight.shape))
    return(weight)

def flatten_except_1dim(weight, ndim=2):
    '''
    change the dimension from:
    (a,b,c,d,..) to (a, b*c*d*..) if ndim = 2
    (a,b,c,d,..) to (a, b*c*d*..,1) if ndim = 3
    '''
    n = weight.shape[0]
    if ndim == 2:
        shape = (n,-1)
    elif ndim == 3:
        shape = (n,-1,1)
    else:
        print("Not implemented!")
    weight = weight.reshape(*shape)
    return(weight)

w_tra = find_weight(y_tra)
w_val = find_weight(y_val)

# w_val = find_weight(y_val)
# w_val_fla = flatten_except_1dim(w_val)
# y_val_fla  = flatten_except_1dim(y_val, ndim=3)

# print("weight_val.shape={}".format(weight_val.shape))
# print("y_val_fla.shape={}".format(y_val_fla.shape))
learning_rate = 1e-4
model = Encoder()
model = model.cuda()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

def load_variable(var, data):
    var.data.resize_(data.shape).copy_(data)

# X_valin = X_val[:32]
# y_valin = y_val[:32]
# load_variable(img_input, torch.FloatTensor(X_valin))
# load_variable(img_label, torch.FloatTensor(y_valin))
# output = model(img_input)

def test_model_for_unittest():
    X_valin = X_val[:32]
    y_valin = y_val[:32]
    load_variable(img_input, X_valin)
    load_variable(img_label, y_valin)
    output = model(img_input)

class facial_keypoint_val_ds(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(X_val)

    def __getitem__(self, index):
        img_np = X_val[index]
        y_np = y_val[index]
        w_np = w_val[index]
        lm = landmark_order
        if random.random() > 0.3:
            img_np = np.transpose(X_val[index], (2, 0, 1))
            y_np = np.transpose(y_val[index], (2, 0, 1))
            w_np = np.transpose(w_val[index], (2, 0, 1))
            return torch.FloatTensor(img_np), torch.FloatTensor(y_np*const), torch.FloatTensor(w_np)
        else:
            lo, ln = np.array(lm["orig"]), np.array(lm["new"])
            '''flipping the img
            '''
            # for i, d in enumerate(img_np):
            #     d = d[:, ::-1,:]
            #     img_np[i] = d
            # img_np = img_np[:, ::-1, :]
            img_np = np.flip(img_np, axis=1)
            '''flipping the img heatmap and img_weights
            '''
            y_orig = y_np[:, :, lo]
            y_np[:, :, lo] = y_np[:, :, ln]
            y_np[:, :, ln] = y_orig
 
            w_orig = w_np[:, :, lo]
            w_np[:, :, lo] = w_np[:, :, ln]
            w_np[:, :, ln] = w_orig
            img_np = np.transpose(X_val[index], (2, 0, 1))
            y_np = np.transpose(y_val[index], (2, 0, 1))
            w_np = np.transpose(w_val[index], (2, 0, 1))
            return torch.FloatTensor(img_np.copy()), torch.FloatTensor(y_np*const),\
                torch.FloatTensor(w_np)

class facial_keypoint_ds(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(X_tra)

    def __getitem__(self, index):
        '''
        there is a chance that a horizontal_flip is used here
        '''
        img_np = X_tra[index]
        y_np = y_tra[index]
        w_np = w_tra[index]
        lm = landmark_order
        if random.random() > 0.3:
            img_np = np.transpose(X_tra[index], (2, 0, 1))
            y_np = np.transpose(y_tra[index], (2, 0, 1))
            w_np = np.transpose(w_tra[index], (2, 0, 1))
            return torch.FloatTensor(img_np), torch.FloatTensor(y_np*const), torch.FloatTensor(w_np)
        else:
            lo, ln = np.array(lm["orig"]), np.array(lm["new"])
            '''flipping the img
            '''
            # for i, d in enumerate(img_np):
            #     d = d[:, ::-1,:]
            #     img_np[i] = d
            # img_np = img_np[:, ::-1, :]
            img_np = np.flip(img_np, axis=1)
            '''flipping the img heatmap and img_weights
            '''
            y_orig = y_np[:, :, lo]
            y_np[:, :, lo] = y_np[:, :, ln]
            y_np[:, :, ln] = y_orig
 
            w_orig = w_np[:, :, lo]
            w_np[:, :, lo] = w_np[:, :, ln]
            w_np[:, :, ln] = w_orig
            img_np = np.transpose(X_tra[index], (2, 0, 1))
            y_np = np.transpose(y_tra[index], (2, 0, 1))
            w_np = np.transpose(w_tra[index], (2, 0, 1))
            return torch.FloatTensor(img_np.copy()), torch.FloatTensor(y_np*const), \
                torch.FloatTensor(w_np)

        return X_val[index], y_val[index], weight_val[index]

skip_train = False
if os.path.exists("model.pth"):
    print("loading model.path skip  training")
    model.load_state_dict(torch.load("model.pth"))
    skip_train = True

train_set = facial_keypoint_ds()
train_loader = data.DataLoader(train_set, batch_size, num_workers=2, shuffle=True)
test_set = facial_keypoint_val_ds()
test_loader = data.DataLoader(test_set, batch_size, num_workers=2, shuffle=True)
show_every = 100
val_every = 1000
val_iter = 100
cur_iter = 0
print("total batches {}".format(len(train_loader)))

if not skip_train:
    for iepoch in range(nb_epochs):
        start = time.time()
        # x_batch, y_batch, w_batch = transform_imgs([X_tra, y_tra, w_tra], landmark_order)
        for x_batch, y_batch, w_batch in iter(train_loader):
            for p in model.parameters():
                p.requires_grad = True
            model.train()
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            w_batch = w_batch.cuda()
            w_batch_fla = flatten_except_1dim(w_batch, ndim=3)
            y_batch_fla = flatten_except_1dim(y_batch, ndim=3)
            optimizer.zero_grad()
            output = model(x_batch)
            masked = output * w_batch_fla
            loss = criterion(masked, y_batch_fla)
            loss.backward()
            optimizer.step()
            history["loss"].append(loss.item())
            # history["val_loss"].append(hist.history["val_loss"][0])
            end = time.time()
            if cur_iter > 0 and cur_iter % show_every == 0:
                print("iter {:03}: loss {:6.4f} {:4.1f}sec".format(
                    cur_iter+1, history["loss"][-1], end-start))
            if cur_iter > 0 and cur_iter % val_every == 0:
                for p in model.parameters():
                    p.requires_grad = False
                model.eval()
                val_losses = []
                for x_val_batch, y_val_batch, w_val_batch in iter(test_loader):
                    x_val_batch = x_val_batch.cuda()
                    y_val_batch = y_val_batch.cuda()
                    w_val_batch = w_val_batch.cuda()
                    w_val_batch_fla = flatten_except_1dim(w_val_batch, ndim=3)
                    y_val_batch_fla = flatten_except_1dim(y_val_batch, ndim=3)
                    val_output = model(x_val_batch)
                    val_masked = val_output * w_val_batch_fla
                    val_loss = criterion(val_masked, y_val_batch_fla)
                    val_losses.append(val_loss.item())
                print("val on iter {}: val loss {:6.4f}".format(
                    iepoch+1, sum(val_losses)/len(val_losses)))
            cur_iter += 1
        print("Epoch {:03}: loss {:6.4f} {:4.1f}sec".format(
            iepoch+1, history["loss"][-1], end-start))

print("training done, storing the model")
torch.save(model.state_dict(), 'model.pth')

def get_ave_xy(hmi, n_points = 4, thresh=0):
    '''
    hmi      : heatmap np array of size (height,width)
    n_points : x,y coordinates corresponding to the top  densities to calculate average (x,y) coordinates
    
    
    convert heatmap to (x,y) coordinate
    x,y coordinates corresponding to the top  densities 
    are used to calculate weighted average of (x,y) coordinates
    the weights are used using heatmap
    
    if the heatmap does not contain the probability > 
    then we assume there is no predicted landmark, and 
    x = -1 and y = -1 are recorded as predicted landmark.
    '''
    if n_points < 1:
        ## Use all
        hsum, n_points = np.sum(hmi), len(hmi.flatten())
        ind_hmi = np.array([range(input_width)]*input_height)
        i1 = np.sum(ind_hmi * hmi)/hsum
        ind_hmi = np.array([range(input_height)]*input_width).T
        i0 = np.sum(ind_hmi * hmi)/hsum
    else:
        ind = hmi.argsort(axis=None)[-n_points:] ## pick the largest n_points
        topind = np.unravel_index(ind, hmi.shape)
        index = np.unravel_index(hmi.argmax(), hmi.shape)
        i0, i1, hsum = 0, 0, 0
        for ind in zip(topind[0],topind[1]):
            h  = hmi[ind[0],ind[1]]
            hsum += h
            i0   += ind[0]*h
            i1   += ind[1]*h

        i0 /= hsum
        i1 /= hsum
    if hsum/n_points <= thresh:
        i0, i1 = -1, -1
    return([i1,i0])

def transfer_xy_coord(hm, n_points = 64, thresh=0.2):
    '''
    hm : np.array of shape (height,width, n-heatmap)
    
    transfer heatmap to (x,y) coordinates
    
    the output contains np.array (Nlandmark * 2,) 
    * 2 for x and y coordinates, containing the landmark location.
    '''
    assert len(hm.shape) == 3
    Nlandmark = hm.shape[-1]
    #est_xy = -1*np.ones(shape = (Nlandmark, 2))
    est_xy = []
    for i in range(Nlandmark):
        hmi = hm[:,:,i]
        est_xy.extend(get_ave_xy(hmi, n_points, thresh))
    return(est_xy) ## (Nlandmark * 2,) 

def transfer_target(y_pred, thresh=0, n_points = 64):
    '''
    y_pred : np.array of the shape (N, height, width, Nlandmark)
    
    output : (N, Nlandmark * 2)
    '''
    y_pred_xy = []
    for i in range(y_pred.shape[0]):
        hm = y_pred[i]
        y_pred_xy.append(transfer_xy_coord(hm, n_points, thresh))
    return(np.array(y_pred_xy))

def gather_xy(pred_xy):
    pred_x = pred_xy[::2]
    pred_y = pred_xy[1::2]
    return pred_x, pred_y

def getRMSE(y_pred_xy, y_train_xy, pick_not_NA):
    res = y_pred_xy[pick_not_NA] - y_train_xy[pick_not_NA]
    RMSE = np.sqrt(np.mean(res**2))
    return(RMSE)

demo_batch = X_val[:200]
img_np = np.transpose(demo_batch, (0, 3, 1, 2))
model.eval()
pred = model(torch.FloatTensor(img_np).cuda())
pred = pred.view(-1, nClasses, output_height, output_width)
pred = pred.permute(0, 2, 3, 1)
y_pred = pred.detach().cpu().numpy()
Nlandmark = pred.size(-1)

for i in range(5):
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(demo_batch[i,:,:,0], cmap="gray")
    pred_xy = transfer_target(y_pred[i][np.newaxis, :, :, :])
    pred_x, pred_y = gather_xy(pred_xy[0])
    gt_xy = transfer_target(y_val[i][np.newaxis, :, :, :])
    gt_x, gt_y = gather_xy(gt_xy[0])
    ax.scatter(pred_x, pred_y, c='red')
    ax.scatter(gt_x, gt_y, c='blue')
    ax.axis("off")

    fig = plt.figure(figsize=(20, 3))
    count = 1
    for j, lab in enumerate(nm_landmarks[::2]):
        ax = fig.add_subplot(2, Nlandmark, count)
        ax.set_title(lab[:10] + "\n" + lab[10:-2])
        ax.axis("off")
        count += 1
        ax.imshow(y_pred[i,:,:,j])
        if j == 0:
            ax.set_ylabel("prediction")

    for j, lab in enumerate(nm_landmarks[::2]):
        ax = fig.add_subplot(2, Nlandmark, count)
        count += 1
        ax.imshow(y_tra[i,:,:,j])
        ax.axis("off")
        if j == 0:
            ax.set_ylabel("true")
    plt.show()

nimage = 200
# y_tra = y_tra[:5]
y_tra = y_val

rmelabels = ["(x,y) from est heatmap  VS (x,y) from true heatmap", 
             "(x,y) from est heatmap  VS true (x,y)             ",
             "(x,y) from true heatmap VS true (x,y)             "]
n_points_width = list(range(1,10))
res = []
n_points_final, min_rmse  = -1 , np.Inf
for nw in  n_points_width + [0]:
    n_points = nw * nw
    y_pred_xy = transfer_target(y_pred[:nimage], 0, n_points)
    y_train_xy = transfer_target(y_tra[:nimage], 0, n_points)
    # print("y_pred_xy {}".format(y_pred_xy))
    # print("y_train_xy {}".format(y_train_xy))
    pick_not_NA = (y_train_xy != -1)
    
    ts = [getRMSE(y_pred_xy, y_train_xy, pick_not_NA)]
    ts.append(getRMSE(y_pred_xy, y_train0.values[:nimage], pick_not_NA))
    ts.append(getRMSE(y_train_xy, y_train0.values[:nimage], pick_not_NA))
    
    res.append(ts)
    
    print("n_points to evaluate (x,y) coordinates = {}".format(n_points))
    print(" RMSE")
    for r, lab in zip(ts,rmelabels):
        print("  {}:{:5.3f}".format(lab,r))
    
    if min_rmse > ts[2]:
        min_rmse = ts[2]
        n_points_final = n_points
        
res = np.array(res)
for i, lab in enumerate(rmelabels):
    plt.plot(n_points_width + [input_width], res[:,i], label = lab)
plt.legend()
plt.ylabel("RMSE")
plt.xlabel("n_points")
plt.show()
