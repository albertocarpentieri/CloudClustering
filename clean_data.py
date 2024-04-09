import numpy as np
import pickle as pkl
import os
import torchvision


folder_path = '/scratch/snx3000/acarpent/train_images/'
output_path = '/scratch/snx3000/acarpent/train_patches/'
img_shape = (1400, 2100)
patch_shape = 256


def create_folder():
    try:
        os.mkdir(output_path)
    except: 
        pass
def main():
    images = os.listdir('/scratch/snx3000/acarpent/train_images/')
    patches = []
    k = 0
    for i in range(len(images)):
        img_tensor = torchvision.io.read_image('/scratch/snx3000/acarpent/train_images/{}'.format(images[i])).numpy().astype(np.float16)
        sum_img_tensor = img_tensor.sum(axis=0)
        img_tensor[:, sum_img_tensor==0] = np.nan
        x = 0
        y = 0
        
        while x <= img_shape[1]-patch_shape:
            while y <= img_shape[0]-patch_shape:
                patch = img_tensor[:, y:y+patch_shape, x:x+patch_shape] 
                if np.isnan(patch).any() == False:
                    patches.append(patch)

                y += patch_shape
            x += patch_shape

        if '{}.pkl'.format(k) not in os.listdir(output_path) and len(patches)>=6:
            with open(output_path+'{}.pkl'.format(k), 'wb') as o:
                pkl.dump(patches[:6], o)
            if len(patches)>6:
                patches = patches[6:]
            else:
                patches = []
            k += 1
            print(k, len(patches))

if __name__ == '__main__':
    main()