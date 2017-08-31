import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from utilities.matrix_dimensions import useful_variables
import pickle as pkl

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (15,14)

def plot_Q(Q_inp,title='Surface_plot.png'):
    actions = np.argmax(Q_inp,2)
    V = np.max(Q_inp,2)
    fig = plt.figure(figsize=plt.figaspect(1.4))
    ax_2 = fig.add_subplot(212)
    ax_2.set_yticks(np.arange(1,22,1))
    ax = fig.add_subplot(211, projection='3d')
    plt.rcParams['figure.figsize'] = (15,12)
    x,y = np.meshgrid(np.arange(10),np.arange(21))
    Z=actions[x,y]
    ax_2.contourf(x+1,y+1,Z.reshape(x.shape),cmap=cm.bone,zorder=5)
    ax_2.xaxis.grid(color='w',linewidth=2,linestyle='dashed',zorder=0)
    ax_2.yaxis.grid(color='w',linewidth=2,linestyle='dashed',zorder=0)
    ax_2.set_axisbelow(False)
    ax.set_yticks(np.arange(1,22,2))
    ax.set_zticks(np.arange(-0.4,1.2,0.2))
    surf = ax.plot_surface(x+1, y+1, V[x,y], cmap=cm.jet,
                           linewidth=0)
    ax.view_init(azim=-13)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig('plots'+'/'+title)
    print(actions)
    return fig

def plot_MSE_lambda(fun,lambs,Q_true,runs,N_0_s,**kwargs):
    '''Args should be 1) function, 2) set of lambdas, 3) the true Q 4) and a list of runs with the number of episodes.
    kwargs are extra arguments for f'''
    fig,ax = plt.subplots()
    if N_0_s:
        colors = plt.cm.rainbow(np.linspace(0,1,len(runs))) if len(N_0_s) == 0 else plt.cm.rainbow(np.linspace(0,1,len(N_0_s)))
        for i,run in enumerate(runs):
            for j,N0 in enumerate(N_0_s):
                MSE_s=[]
                for lambd in lambs:
                    print('Running with {:.2f} lambda and {} N0 over {}'.format(lambd,N0,run))
                    Q_l=fun(lamb=lambd, loops=run, **kwargs)
                    MSE_s+=[np.sum(np.square(Q_l-Q_true))/Q_true.size]
                color = i if len(N_0_s) == 0 else j
                plt.plot(lambs,MSE_s,
                         label=str(N0)+'N0'+' '+str(run)+' '+'episodes',
                         color=colors[color])
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(labels)
        if len(N_0_s) > 0:
            plt.title('MSE over runs of {} on N_0 of {}'.format(','.join([str(x) for x in runs]),
                                                                ','.join([str(x) for x in N_0_s])))
            fig.savefig('plots/MSE_against_lambda_{}_multiple_N0s.png'.format(fun.__name__))
            return fig
        plt.title('MSE over runs of {}'.format(','.join([str(x) for x in runs])))
        fig.savefig('plots/MSE_against_lambda_{}.png'.format(fun.__name__))
    else:
        colors = plt.cm.rainbow(np.linspace(0,1,len(runs)))
        for i,run in enumerate(runs):
            MSE_s=[]
            for lambd in lambs:
                Q_l=fun(lamb=lambd, loops=run, **kwargs)
                MSE_s+=[np.sum(np.square(Q_l-Q_true))/Q_true.size]
            plt.plot(lambs,MSE_s,
                     label=str(run)+' '+'episodes',
                     color=colors[i])
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(labels)
        plt.title('MSE over runs of {}'.format(','.join([str(x) for x in runs])))
        fig.savefig('plots/MSE_against_lambda_{}.png'.format(fun.__name__))
    return fig

def plot_MSE_episodes(f,n_episodes,lambs,**kwargs):
    '''Args should be 1) function, 2) how many episodes to run and 3) a set of lambdas.
    kwargs are extra arguments for f'''
    fig,ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0,1,len(lambs)))
    lines = []
    for i,la in enumerate(lambs):
        print('Plotting MSE against ran of episodes for lambda'+' '+format(la,'.2f'))
        Q_i, MSE_s = f(lamb=la,loops=n_episodes,**kwargs)
        plt.plot(np.arange(1,n_episodes+1,50),MSE_s,color=colors[i],label=format(la,'.2f')+' '+'lambda',alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(labels)
    plt.title('Plot of MSE over 40000 episodes')
    fig.savefig('plots'+'/'+'MSE_episodes_lambda_{}_over_lambdas'.format(f.__name__))
    return fig
