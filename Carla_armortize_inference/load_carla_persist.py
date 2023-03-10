import glob, os
import numpy as np
from PIL import Image
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import Resize, ToTensor, Compose

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

class ImageDataset(VisionDataset):
    """
    Load images from multiple data directories.
    Folder structure: data_dir/filename.png
    """

    def __init__(self, data_dirs, fov=30, radius=10, im_sz=128):
        # Use multiple root folders
        self.data_dir = data_dirs
        if not isinstance(data_dirs, list):
            data_dirs = [os.path.join(data_dirs, 'imgs')]

        transforms = Compose([
            Resize(im_sz),
            ToTensor(),
        ])

        # initialize base class
        VisionDataset.__init__(self, root=data_dirs, transform=transforms)

        self.filenames = []
        root = []

        for ddir in self.root:
            filenames = self._get_files(ddir)
            self.filenames.extend(filenames)
            root.append(ddir)
        self.im_sz = im_sz
        self.fov = fov
        self.radius = radius
        self.focal = .5 * im_sz / np.tan(.5 * fov / 180 * np.pi)

        theta = np.linspace(-180, 180, 10)
        phi = np.linspace(-85, -15, 10)
        render_poses = []
        for t in theta:
            for p in phi:
                render_poses.append(pose_spherical(t, p, self.radius))
        self.render_poses = torch.stack(render_poses, 0)


    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def _get_files(root_dir):
        return glob.glob(f'{root_dir}/*.png') + glob.glob(f'{root_dir}/*.jpg')

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        posefile = os.path.join(self.data_dir, 'poses', filename[-10:-4] + '_extrinsics.npy')
        pose = torch.Tensor(np.concatenate([np.load(posefile), np.array([[0, 0, 0, 1]])], axis=0))

        return img, pose, torch.tensor([idx], dtype=torch.long)
    
    def sample_pose_matrix(self):
        #t = np.random.uniform(-180, 180)
        #p = np.random.uniform(-85, -15)
        t = np.random.uniform(-180, 180)
        v_min = 0.0
        v_max = 0.45642212862617093
        v = np.random.uniform(low=v_min, high=v_max)
        p = -1.0 * (90 - np.arccos(1 - 2 * v) / np.pi * 180)
        
        pose = pose_spherical(t, p, self.radius)
        return pose



