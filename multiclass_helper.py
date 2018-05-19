from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def get_custom_cmap(Ri=0, Gi=0, Bi=0, alpha=0.8):
    N_c = 256
    R = np.hstack([np.zeros(int(N_c/2)), np.linspace(1, Ri, 128)])
    G = np.hstack([np.zeros(int(N_c/2)), np.linspace(1, Gi, 128)])
    B = np.hstack([np.zeros(int(N_c/2)), np.linspace(1, Bi, 128)])
    A = np.hstack([np.zeros(int(N_c/2)), alpha*np.ones(int(N_c/2))])
    custom_map = ListedColormap(np.vstack([R,G,B,A]).T)
    return custom_map

def plot_MC_boundaries_keras(X_train, y_train, score, probability_func, degree=None, bias=False, h = .02, ax = None, margin=0.5):
    y_train_cat = to_categorical(y_train)
    X = X_train
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    if ax is None:
        ax = plt.subplot(1, 1, 1)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    if degree is not None:
        polynomial_set = get_polynimial_set(np.c_[xx.ravel(), yy.ravel()], degree = degree, bias=bias)
        Zaux = probability_func(polynomial_set)
    else:
        Zaux = probability_func(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z_aux[:, 1]
    
    # if Zaux.shape[1] == 2:
        # Es un polinomio
        # Z = Zaux[:, 1]
    # else:
        # No es un polinomio
        # Z = Zaux[:, 2]

    # Put the result into a color plot
    Z_reshaped = Zaux.reshape(xx.shape[0], xx.shape[1], y_train_cat.shape[1])
    
    cm_borders = ListedColormap(["#FFFFFFFF", "#000000"])
    my_colors = [[0,0,0.5], [0,0.5,0], [0.5,0,0], [0,0,0], [0,0.5,0.5]]
    for i in range(Z_reshaped.shape[2]):
        my_cmap = get_custom_cmap(my_colors[i][0],my_colors[i][1],my_colors[i][2], 0.5)
        Z = Z_reshaped[:,:,i]    
        
        cf = ax.contourf(xx, yy, 
                         #Z*(Z>0.5) + (Z<0.5)*0.5, 
                         Z,
                         50, 
                         vmin = 0,
                         vmax = 1,
                         cmap=my_cmap, 
                         #alpha=.9
                        )
        
    ax.scatter(X_train[:, 0], X_train[:, 1], 
               c=y_train, 
               cmap=ListedColormap(my_colors),
               edgecolors='k', 
               s=100)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=40, horizontalalignment='right')
    return Zaux, Z

def generate_dataset(random_variables):
    X = np.array([]).reshape(0, len(random_variables[0][0]))
    y = np.array([]).reshape(0, 1)
    for i, rv in enumerate(random_variables):
        X = np.vstack([X, np.random.multivariate_normal(rv[0], rv[1], rv[2])])
        y = np.vstack([y, np.ones(rv[2]).reshape(rv[2],1)*i]) 
    y = y.reshape(-1)
    return X, y