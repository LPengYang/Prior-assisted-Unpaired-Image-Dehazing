import torch
import torch.optim as optim
from torchvision import  transforms
import os
import time
from torch.optim import lr_scheduler
from tqdm import tqdm
from utilities import Transmission_estimator, Dehaze_mix,GANLoss,DCPDehazeGenerator,ImagePool
import utilities
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument(
        '--save_dir',         
        type=str,             
        default='',    
    ) 
    parser.add_argument(
        '--train_haze_dir',
        type=str,
        default='',
    )
    parser.add_argument(
        '--train_clean_dir',
        type=str,
        default='',
    )
    args = parser.parse_args()
    
    gpu_id = 0
    device = torch.device(gpu_id)
    torch.cuda.set_device(gpu_id)
    utilities.setup_seed(2023)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    haze_dir = args.train_haze_dir
    clear_dir = args.train_clean_dir
    path_save = args.save_dir
    
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    batch_number = 4
    confident_dcp_value = 0.8
    hyperparas = [0.1, 0.2, 0.4]

    total_epoch = 100
    redensity_min = 1.5
    redensity_max = 3
    images_pool_number = 200
    learn_lr = 0.0002

    dehaze_t = Transmission_estimator(output_nc = 2).to(device)
    dehaze_j = utilities.init_net(Dehaze_mix(input_nc=3,output_nc=3,ngf=64,n_downsampling=2, n_mixblock=1,)).to(device)
    net_dis  = utilities.define_D(input_nc=3,ndf=64).to(device)
    Loss_gan = GANLoss(gan_mode='lsgan').to(device)

    optimizer_gen = optim.Adam([
        {'params': dehaze_t.parameters()},
        {'params': dehaze_j.parameters()},
        ], lr=learn_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer_dis = optim.Adam([
        {'params': net_dis.parameters()},
        ], lr=learn_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    data_transform = transforms.Compose([  
    transforms.RandomResizedCrop((256,256),scale=(0.8,1),ratio=(1.0,1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    img_loader = utilities.Unpaired_Dataset(data_transform,haze_dir,clear_dir)
    total_image_number = max(img_loader.I_size,img_loader.J_size)
    imgLoader = torch.utils.data.DataLoader(img_loader, batch_size=batch_number, shuffle=True, num_workers=32)

    DCPModule = DCPDehazeGenerator(win_size = 15).to(device)
    Loss_l1 = torch.nn.L1Loss().to(device)
    fake_pool_1, fake_pool_2 = ImagePool(images_pool_number), ImagePool(images_pool_number)

    epoch = 0
    confident_dcp  = torch.tensor(confident_dcp_value).to(device)
    
    while epoch < total_epoch:  
        epoch += 1
        start_time =time.time() 
        pbar = tqdm(total=total_image_number//batch_number, desc='Train',colour='cyan')

        for i, (haze,clear) in enumerate(imgLoader, 0):
            haze  = haze.to(device)
            clear = clear.to(device)
            
            result_t = dehaze_t(haze).clamp(0.001,1)
            haze_t = result_t[:,[0],:,:]
            confident_t = result_t[:,[1],:,:]
            shape = haze.shape

            # breakpoint()
            _, t_dcp, map_A = DCPModule(haze) 

            image_dehaze = dehaze_j(haze)

            loss_prior = (confident_t*abs(haze_t-t_dcp)).mean() + Loss_l1(confident_t.mean(), confident_dcp)

            dehaze_asm = utilities.dehaze_fog(haze,haze_t,map_A)
            haze_syn = utilities.synthesize_fog(image_dehaze,haze_t,map_A)
            loss_rec =  Loss_l1(haze_syn,haze) + Loss_l1(image_dehaze,dehaze_asm)
            loss_gan_gen = Loss_gan(net_dis(image_dehaze),True) 

            re_density = redensity_min+(redensity_max-redensity_min)*torch.rand(1,device=device)
            t_rehaze = torch.exp(torch.log(haze_t)*re_density)

            image_rehaze_t = utilities.synthesize_fog(image_dehaze,t_rehaze,map_A)
            
            haze_dehaze_pair = fake_pool_2.query(torch.cat((image_rehaze_t,image_dehaze),dim=1))
            rehaze_sample, dehaze_sample = haze_dehaze_pair[:,:3,:,:].detach().clamp(-1,1), haze_dehaze_pair[:,3:,:,:].detach()
            dehaze_rehaze = dehaze_j(rehaze_sample)
            loss_hazemodified =  Loss_l1(dehaze_rehaze,dehaze_sample) 

            loss = loss_rec + hyperparas[0]*loss_gan_gen + hyperparas[1]*loss_prior  + hyperparas[2]*loss_hazemodified 

            utilities.set_requires_grad(net_dis,False)
            optimizer_gen.zero_grad()
            loss.backward()
            optimizer_gen.step()
            utilities.set_requires_grad(net_dis,True)

            fake_dehaze = fake_pool_1.query(image_dehaze)
            loss_dis = (Loss_gan(net_dis(fake_dehaze.detach()),False)+Loss_gan(net_dis(clear),True))/2
            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()

            pbar.update(1)
            pbar.set_postfix({"G": f"{loss_gan_gen.item():.2f}", "R": f"{loss_rec.item():.2f}", 
                                "P": f"{loss_prior.item():.2f}", "C": f"{loss_hazemodified.item():.2f}"})
        
        print('In epoch: %4d loss_gan_gen: %0.2f loss_dis_gen: %0.2f loss_rec: %0.2f loss_prior: %0.2f loss_hm: %0.2f '\
                        %(epoch, loss_gan_gen.item(), loss_dis.item(), loss_rec.item(), loss_prior.item(), loss_hazemodified.item(),))
                        
        pbar.close()    
        if (epoch+1)%10==0:
            torch.save(dehaze_t.state_dict(), os.path.join(path_save, str(epoch+1)+'_epoch_dehaze_t.pt'))
            torch.save(dehaze_j.state_dict(), os.path.join(path_save, str(epoch+1)+'_epoch_dehaze_j.pt'))
            torch.save(net_dis.state_dict(), os.path.join(path_save, str(epoch+1)+'_epoch_dis.pt'))
            torch.save(optimizer_gen.state_dict(), os.path.join(path_save, str(epoch+1)+'_epoch_optimizer_gen.pt'))
            torch.save(optimizer_dis.state_dict(), os.path.join(path_save, str(epoch+1)+'_epoch_optimizer_dis.pt'))
        
        torch.save(dehaze_t.state_dict(), os.path.join(path_save,'current_dehaze_t.pt'))
        torch.save(dehaze_j.state_dict(), os.path.join(path_save,'current_dehaze_j.pt'))
        torch.save(net_dis.state_dict(), os.path.join(path_save,'current_dis.pt'))
        torch.save(optimizer_gen.state_dict(), os.path.join(path_save, 'current_optimizer_gen.pt'))
        torch.save(optimizer_dis.state_dict(), os.path.join(path_save, 'current_optimizer_dis.pt'))

        print('end_epoch:',epoch, 'lr:', optimizer_gen.state_dict()['param_groups'][0]['lr'])