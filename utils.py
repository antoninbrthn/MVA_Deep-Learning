import matplotlib.pyplot as plt
import numpy as np
import keras
from mp1 import *
from math import *


def visualize_prediction_mult(x_tab, y_tab, n = 10, size = (10, 10), ordered = True, show_order = True, title =None):
    k = int(floor(sqrt(n)))
    fig, ax = plt.subplots(nrows = k, ncols = k, figsize=size)
    plt.suptitle(title, size=20)
    if ordered:
        index = np.arange(0, n)
    else:
        index = np.random.choice(np.arange(0, x_tab.shape[0]), size = k**2)
    for i in range(k):
        for j in range(k):
            x,y = x_tab[index[i*k + j]], y_tab[index[i*k + j]]
            I = x.reshape((IMAGE_SIZE,IMAGE_SIZE))
            ax[i,j].imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
            ax[i,j].set_xlim([0,1])
            ax[i,j].set_ylim([0,1])

            xy = y.reshape(3,2)
            tri = patches.Polygon(xy, closed=True, fill = False, edgecolor = 'r', linewidth = 3, alpha = 0.6)
            ax[i, j].add_patch(tri)
            
            if show_order:
                for s in range(3):
                    ax[i,j].annotate(str(s), xy=(y[s]), xytext=(y[s]), color='y', size=20)
    plt.show()
    

def get_worst(X, Y, Y_pred, N):
    scores = np.array([np.linalg.norm(y_true - y_pred) for y_true,y_pred in zip(Y, Y_pred)])
    return np.argsort(scores)[-N:]


def get_best(X, Y, Y_pred, N):
    scores = [np.linalg.norm(y_true - y_pred) for y_true,y_pred in zip(Y, Y_pred)]
    return np.argsort(scores)[:N]

def get_prediction_score(prediction, y):
    bools = np.argmax(prediction,axis=1) == np.argmax(y, axis=1)
    return sum(bools)/len(bools)



#############################
## Predict triangle vertices
#############################
def order_array(data):
    Y = data.copy()
    s = Y[:,:, 0].argsort(axis=1)
    for i in range(Y.shape[0]):
        Y[i] = Y[i, s[i]]
    return Y

def second_order_array(data, threshold = 0.12):
    couples = [[0, 1], [1, 2], [0, 2]]
    y = data.copy()
    for i in range(y.shape[0]):
        for c in couples:
            if abs(y[i,:,0][c][0] - y[i,:,0][c][1]) < threshold:
                if y[i,:, 1][c][0] < y[i,:, 1][c][1]:
                    temp = y[i,:, 1][c[0]]
                    y[i,:, 1][c[0]] = y[i,:, 1][c[1]]
                    y[i,:, 1][c[1]] = temp
                    break
    return y

def normalize_y_origin(Y):
    sort_y = lambda y: y[np.argsort(np.linalg.norm(y, axis=1))]
    for i in range(Y.shape[0]):
        Y[i] = sort_y(Y[i])
    return Y

def normalize_y_center(Y):
    center = np.array([0.5, 0.5])
    sort_y = lambda y: y[np.argsort(np.linalg.norm(y-center, axis=1))]
    for i in range(Y.shape[0]):
        Y[i] = sort_y(Y[i])
    return Y

def normalize_y_diag(data):
    Y = data.copy()
    sort_y = lambda y: y[np.argsort(np.linalg.norm(np.array([1, 1]) - y, axis=1)+np.linalg.norm(y, axis=1))]
    for i in range(Y.shape[0]):
        Y[i] = sort_y(Y[i])
    return Y

def get_angle(a, origin = np.array([0, 1])):
    center = np.array([0.5, 0.5])
    b = (a - center)/np.linalg.norm(a - center, axis=1).reshape(3, 1)
    signe = -np.sign(b[:,0])
    cst = np.array([0 if s == 1 else 2 for s in signe])
    acos_orig = lambda x: acos(np.dot(x, origin))
    return signe * np.array(list(map(acos_orig, b)))/pi + cst

def get_angle_bary(a, origin = np.array([0, 1])):
    '''Get angle wrt barycenter of the triangle'''
    center = np.mean(a, axis=0)
    b = (a - center)/np.linalg.norm(a - center, axis=1).reshape(3, 1)
    signe = -np.sign(b[:,0])
    cst = np.array([0 if s == 1 else 2 for s in signe])
    acos_orig = lambda x: acos(np.dot(x, origin))
    return signe * np.array(list(map(acos_orig, b)))/pi + cst

def trigo_sort(data):
    y = data.copy()
    for i in range(y.shape[0]):
        sort_i = np.argsort(get_angle(y[i]))
        y[i] = y[i][sort_i]
    return y

def trigo_sort_barycenter(data):
    y = data.copy()
    for i in range(y.shape[0]):
        sort_i = np.argsort(get_angle_bary(y[i]))
        y[i] = y[i][sort_i]
    return y

def trigo_sort_barycenter_shift(data, beta=0.1):
    y = data.copy()
    for i in range(y.shape[0]):
        sort_i = np.argsort(get_angle_bary_shift(y[i], beta=0.1))
        y[i] = y[i][sort_i]
    return y

def get_angle_bary_shift(a, origin = np.array([0, 1]), beta=0.1):
    '''Get angle wrt shifted barycenter of the triangle '''
    center = get_barycenter_shift(a, beta = 0.1)
    b = (a - center)/np.linalg.norm(a - center, axis=1).reshape(3, 1)
    signe = -np.sign(b[:,0])
    cst = np.array([0 if s == 1 else 2 for s in signe])
    acos_orig = lambda x: acos(np.dot(x, origin))
    return signe * np.array(list(map(acos_orig, b)))/pi + cst

def get_barycenter_shift(a, origin=np.array([0,1]), beta = 0.1):
    center = np.mean(a, axis=0)
    shift = 0
    for x in a:
        # alignment coefficent with the y axis
        align_coef = np.dot((x-center)/np.linalg.norm((x-center)), origin)
        shift += beta*exp(10*(align_coef-1))
    center[0] += shift
    return center


def visualize_prediction_with_order(x, y):
    fig, ax = plt.subplots(figsize=(5, 5))
    I = x.reshape((IMAGE_SIZE,IMAGE_SIZE))
    ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    xy = y.reshape(3,2)
    tri = patches.Polygon(xy, closed=True, fill = False, edgecolor = 'r', linewidth = 3, alpha = 0.5)
    for i in range(3):
        ax.annotate(str(i), xy=(y[i]), xytext=(y[i]), color='y', size=20)
    ax.add_patch(tri)

    plt.show()
    

def plot_history(history):
    plt.figure(figsize=(7, 5))
    plt.title("Evolution of the loss")
    plt.plot(history.history['loss'], label = "Training data")
    plt.plot(history.history['val_loss'], label = "Validation data")
    plt.legend()
    plt.show()

#############################
## Denoising hourglass
#############################
def generate_dataset_denoising(nb_samples, max_noise=0.2, free_location=False):
    # Getting im_size:
    im_size = generate_a_rectangle().shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros([nb_samples,im_size])
    # print('Creating data:')
    for i in range(nb_samples):
        noise = np.random.rand()*max_noise
        # if i % 10 == 0:
        #     print(i)
        category = np.random.randint(3)
        if category == 0:
            img_pair = generate_a_rectangle(noise, free_location, paired = True)
            X[i], Y[i] = img_pair[0], img_pair[1]
        elif category == 1: 
            img_pair = generate_a_disk(noise, free_location, paired = True)
            X[i], Y[i] = img_pair[0], img_pair[1]
        else:
            img_pair = generate_a_triangle(noise, free_location, paired = True)
            X[i], Y[i] = img_pair[0], img_pair[1]
    X = (X + noise) / (255 + 2 * noise)
    Y = Y / 255
    return [X, Y]

def plot_pair(x, y, size= (10, 5), n=72):
    plt.figure(figsize = size)
    plt.subplot(1, 2, 1)
    plt.imshow(x.reshape(n,n), cmap="binary")
    plt.subplot(1, 2, 2)
    plt.imshow(y.reshape(n,n), cmap="binary")
    plt.show()
