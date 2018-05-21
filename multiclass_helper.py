from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def get_custom_cmap(Ri=0, Gi=0, Bi=0, alpha=0.8):
    N_c = 256
    N_c_col = int(N_c/2)
    marg = 0
    R = np.hstack([np.zeros(N_c_col-marg), np.linspace(1, Ri, N_c_col+marg)])
    G = np.hstack([np.zeros(N_c_col-marg), np.linspace(1, Gi, N_c_col+marg)])
    B = np.hstack([np.zeros(N_c_col-marg), np.linspace(1, Bi, N_c_col+marg)])
    A = np.hstack([np.zeros(N_c_col-marg), alpha*np.ones(N_c_col+marg)])
    custom_map = ListedColormap(np.vstack([R,G,B,A]).T)
    return custom_map

def plot_MC_boundaries_keras(X_train, y_train, score, probability_func, degree=None, bias=False, mesh_res = 300, h = .02, ax = None, margin=0.5, color_index = 0, normalize = False):
    y_train_cat_aux = to_categorical(y_train)
    if (y_train_cat_aux.shape[1] > 2):
        y_train_cat = y_train_cat_aux
    else:
        y_train_cat = y_train
    X = X_train
    margin_x = (X[:, 0].max() - X[:, 0].min())*0.05
    margin_y = (X[:, 1].max() - X[:, 1].min())*0.05
    x_min, x_max = X[:, 0].min() - margin_x, X[:, 0].max() + margin_x
    y_min, y_max = X[:, 1].min() - margin_y, X[:, 1].max() + margin_y
    hx = (x_max-x_min)/mesh_res
    hy = (y_max-y_min)/mesh_res
    x_domain = np.arange(x_min, x_max, hx)
    y_domain = np.arange(y_min, y_max, hy)
    xx, yy = np.meshgrid(x_domain, y_domain)
    
    
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
    
    if normalize:
        Zaux = (Zaux.T/Zaux.sum(axis=1)).T
    
    cm_borders = ListedColormap(["#FFFFFFFF", "#000000"])
    my_colors = [[0,0,0.5], [0,0.5,0], [0.5,0,0], [0,0,0], [0,0.5,0.5]]
    
    cat_order = len(y_train_cat.shape)
    if cat_order>1:
        Z_reshaped = Zaux.reshape(xx.shape[0], xx.shape[1], y_train_cat.shape[1])
        for i in range(Z_reshaped.shape[2]):
            my_cmap = get_custom_cmap(my_colors[i][0],my_colors[i][1],my_colors[i][2], alpha=0.5)
            Z = Z_reshaped[:,:,i]    

            cf = ax.contourf(xx, yy,
                             Z,
                             50, 
                             vmin = 0,
                             vmax = 1,
                             cmap=my_cmap, 
                            )
            ax.scatter(X_train[:, 0], X_train[:, 1], 
               c=y_train, 
               cmap=ListedColormap(my_colors),
               edgecolors='k', 
               s=100)
    else:
        Z_reshaped = Zaux.reshape(xx.shape[0], xx.shape[1])
        my_cmap = get_custom_cmap(my_colors[color_index][0],my_colors[color_index][1],my_colors[color_index][2], alpha=0.5)
        cf = ax.contourf(xx, yy,
                             Z_reshaped,
                             50, 
                             vmin = 0,
                             vmax = 1,
                             cmap=my_cmap, 
                            )
        ax.scatter(X_train[:, 0], X_train[:, 1], 
               c=y_train, 
               # cmap=ListedColormap(my_colors[color_index]),
               edgecolors='k', 
               s=100)
    
        
    boundary_line = np.where(np.abs(Z_reshaped-0.5)<0.001)
    
    ax.scatter(x_domain[boundary_line[1]], y_domain[boundary_line[0]], color='k', alpha=0.5, s=1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #ax.set_xticks(())
    #ax.set_yticks(())
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=40, horizontalalignment='right')
    #return Zaux

def generate_dataset(random_variables):
    X = np.array([]).reshape(0, len(random_variables[0][0]))
    y = np.array([]).reshape(0, 1)
    for i, rv in enumerate(random_variables):
        X = np.vstack([X, np.random.multivariate_normal(rv[0], rv[1], rv[2])])
        y = np.vstack([y, np.ones(rv[2]).reshape(rv[2],1)*i]) 
    y = y.reshape(-1)
    return X, y