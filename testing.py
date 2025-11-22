import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.utils as vutils
import torch.nn.functional as F
from utilities import Dehaze_mix,EvalDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument(
        '--model_load_path',         
        type=str,             
        default='',    
    ) 
    parser.add_argument(
        '--testing_image_dir',
        type=str,
        default='',
    )
    parser.add_argument(
        '--save_results_dir',
        type=str,
        default='',
    )
    args = parser.parse_args()
    
    gpu_id = 0
    device = torch.device(gpu_id)
    torch.cuda.set_device(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
   
    path_load = args.model_load_path
    hazy_image_dir = args.testing_image_dir
    path_save_images = args.save_results_dir
    
    if not os.path.exists(path_save_images):
        os.mkdir(path_save_images)

    dehaze_j = Dehaze_mix(input_nc=3,output_nc=3,ngf=64,n_downsampling=2, n_mixblock=1,).to(device)
    dehaze_j.load_state_dict(torch.load(path_load, map_location='cuda:0'))
    dehaze_j.eval()

    data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    img = EvalDataset(hazy_image_dir, data_transform)
    imgLoader = torch.utils.data.DataLoader(img, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for i, (haze,haze_name) in enumerate(imgLoader, 0):
            image = haze.to(device)
            shape = image.shape
            image = F.interpolate(image, size=(shape[2]-shape[2]%8,shape[3]-shape[3]%8) , mode="bilinear", align_corners=True)

            image_dehaze = dehaze_j(image)
            image_dehaze = F.interpolate(image_dehaze, size=(shape[2],shape[3]), mode="bilinear", align_corners=True)
            vutils.save_image(image_dehaze[0,:]/2+0.5,path_save_images+haze_name[0])
