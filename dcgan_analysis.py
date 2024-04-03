from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dcgan import Generator
import torch
import skgstat as skg

def main():
    checkpoint = 10
    checkpoint_path = "test_checkpoints/checkpoint_epoch_10.pth"

    sgs_df = pd.read_parquet("GL_ensemble.parquet.gzip", engine = "fastparquet")
    dcgan_df = dcgan_sim(checkpoint_path, 25)
   
    print("Begin")

    coords = sgs_df[['X','Y']].values.reshape(100,100,2)
    coords = coords[:96,:96].reshape(9216,2)

    sgs_topo = load_sgs_sim(sgs_df, 25)

    for i in range(25):
        ele = dcgan_df[:,i]
        norm_ele = (ele - 0.5) * 2
        dcgan_df[:,i] = norm_ele
    
    # Variance of dcgan samples at each location
    dcgan_var = np.mean(dcgan_df, axis=0)
    print(dcgan_var.shape)

    # Variance of SGS samples at each location
    SGS_var = np.mean(sgs_topo, axis=0)
    print(SGS_var.shape)

    # Variance of SGS data
    plot_colormap(coords[:,0], coords[:,1], SGS_var, "Normalized Mean Ele", 
                       "SGS Elevation", checkpoint, "SGS_mean.png")

    # Variance of DCGAN data
    plot_colormap(coords[:,0], coords[:,1], dcgan_var, "Normalized Mean Ele", 
                   f"DCGAN Elevation CP {checkpoint}", checkpoint, "DCGAN_mean.png")

    # Difference map of Variance
    plot_colormap(coords[:,0], coords[:,1],SGS_var - dcgan_var, "Difference Norm Mean Ele", 
                    f"SGS - DCGAN Elevation CP {checkpoint}", checkpoint, "diff_mean.png")
    
    plot_variograms(coords, sgs_topo, dcgan_df, checkpoint)

    print("done")


def dcgan_sim(checkpoint_path, num_images):
    generator = Generator(96, 100, 1) # default parameters
    generator.eval()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    generator.load_state_dict(checkpoint['generator_state_dict'])

    z = torch.randn(num_images, 100) 
    with torch.no_grad():
        generated_images = generator(z)
    
    return generated_images.cpu().numpy().reshape(num_images, -1)

def load_sgs_sim(img_df, num_images):

    image_df = pd.read_parquet("GL_ensemble.parquet.gzip", engine = "fastparquet")
    image_names = [f'sim_{i}' for i in range(1,4041)]

    rand_sim = np.random.randint(0, high=4041, size=num_images)

    tmp = np.zeros((num_images,100,100), dtype='float')
    sgs_sim = np.zeros((num_images,9216), dtype='float')

    for i, n in enumerate(rand_sim):
        tmp[i] = np.array(img_df[image_names[n]].values).reshape(100,100)
        sgs_sim[i] = tmp[i,:96,:96].reshape(9216)

    return sgs_sim

def plot_colormap(X, Y, Color, colorbar_title, title, cp, filename):

    fig = plt.figure(figsize = (6,5))
    ax = plt.gca()
    im = ax.scatter(X, Y, c=Color, marker='.', s=5, cmap='gist_earth')
    plt.title(title)
    plt.xlabel('X [m]'); plt.ylabel('Y [m]')
    plt.locator_params(nbins=5)
    plt.axis('scaled')

    # make colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, ticks=np.linspace(np.min(Color), np.max(Color), 11), cax=cax)
    cbar.set_label(colorbar_title, rotation=270, labelpad=15)
    save_dir = f'compare/cp_{cp}'
    os.makedirs(save_dir, exist_ok=True)
    
    fig.savefig(os.path.join(save_dir, filename), dpi=fig.dpi)

def plot_variograms(coords, sgs, dcgan, cp):

    maxlag = 50000      # maximum range distance
    n_lags = 70         # num of bins

    save_dir = f'compare/cp_{cp}'
    os.makedirs(save_dir, exist_ok=True)

    for i in range(8):

        print("Starting variogram calculation")
        V1 = skg.Variogram(coords, sgs[i], bin_func='even', n_lags=n_lags, maxlag=maxlag, normalize=False)
        print(f"Finished w/ Variogram {i*4+1}\n")
        V2 = skg.Variogram(coords, dcgan[i*3], bin_func='even', n_lags=n_lags, maxlag=maxlag, normalize=False)
        print(f"Finished w/ Variogram {i*4+2}\n")
        V3 = skg.Variogram(coords, dcgan[i*3+1], bin_func='even', n_lags=n_lags, maxlag=maxlag, normalize=False)
        print(f"Finished w/ Variogram {i*4+3}\n")
        V4 = skg.Variogram(coords, dcgan[i*3+2], bin_func='even', n_lags=n_lags, maxlag=maxlag, normalize=False)
        print(f"Finished w/ Variogram {i*4+4}\n")


        xdata = V1.bins
        V1data = V1.experimental
        V2data = V2.experimental
        V3data = V3.experimental
        V4data = V4.experimental

        fig = plt.figure(figsize=(6,4))
        plt.scatter(xdata, V1data, s=12, c='k', marker='*', label='SGS Sim')
        plt.scatter(xdata, V2data, s=12, c='r', label=f'DCGAN Sim {i*3+1}')
        plt.scatter(xdata, V3data, s=12, c='g', label=f'DCGAN Sim {i*3+2}')
        plt.scatter(xdata, V4data, s=12, c='b', label=f'DCGAN Sim {i*3+3}')
        plt.legend(loc="upper left")
        plt.title(f'Isotropic Experimental Variogram CP {cp}')
        plt.xlabel('Lag (m)'); plt.ylabel('Semivariance')
        fig.savefig(os.path.join(save_dir, f'vario_sgs_DCGAN{i*3+1}-{i*3+3}.png'), dpi=fig.dpi)

if __name__ == "__main__":
    main()