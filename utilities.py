import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vgg19_bn,VGG19_BN_Weights
from torch.nn import init
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import random
import os

# Guided image filtering for grayscale images
class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3, gpu_ids=None):    # only work for gpu case at this moment
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        self.boxfilter = nn.AvgPool2d(kernel_size=2*self.r+1, stride=1,padding=self.r)

    def forward(self, I, p):
        """
        I -- guidance image, should be [0, 1]
        p -- filtering input image, should be [0, 1]
        """
        
        # N = self.boxfilter(self.tensor(p.size()).fill_(1))
        N = self.boxfilter( torch.ones(p.size()) )

        if I.is_cuda:
            N = N.cuda()

        # print(N.shape)
        # print(I.shape)
        # print('-----------')

        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I*p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(I*I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        return mean_a * I + mean_b
    
class DCPDehazeGenerator(nn.Module):
    """Create a DCP Dehaze generator"""
    def __init__(self, win_size= 15,  eps=1e-3):
        super(DCPDehazeGenerator, self).__init__()

        self.guided_filter = GuidedFilter(r=win_size, eps=eps)
        self.win_size = win_size
        self.omega = 0.95
        self.maxpool = torch.nn.MaxPool2d(kernel_size=win_size,stride=1)


    def get_dark_channel(self,x):
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = F.pad(x, (self.win_size //2, self.win_size //2,self.win_size //2, self.win_size//2), mode='constant', value=1)
        x = -(self.maxpool(-x))
        return x
    
    def atmospheric_light(self,x):
        intensity_image = 0.2989 * x[:,0,:,:] + 0.5870 * x[:,1,:,:] + 0.1140 * x[:,2,:,:]
        dark_channel = self.get_dark_channel(x)
        searchidx = torch.argsort(dark_channel.view(x.shape[0],-1), dim=1, descending=True)[:,:int(x.shape[2]*x.shape[3]*0.001)]
        x_reshape = x.view(x.shape[0],3,-1)
        searched = torch.gather(x_reshape,dim=2,index=searchidx.unsqueeze(1).repeat(1,3,1))
        intensity_idx = torch.argsort(torch.gather(intensity_image.view(x.shape[0],-1),dim=1,index=searchidx),dim=1,descending=True)[:,0] 
        A_final_pixel = torch.gather(searched,dim=2,index = intensity_idx.unsqueeze(1).unsqueeze(1).repeat(1,3,1))
        
        map_A = A_final_pixel.unsqueeze(3).repeat(1,1,x.shape[2],x.shape[3])

        return map_A 
       

    def forward(self, x):
        if x.shape[1] > 1:
            # rgb2gray
            guidance = 0.2989 * x[:,0,:,:] + 0.5870 * x[:,1,:,:] + 0.1140 * x[:,2,:,:]
        else:
            guidance = x
        # rescale to [0,1]
        guidance = (guidance + 1)/2
        guidance = torch.unsqueeze(guidance, dim=1)
        imgPatch = (x + 1)/2

        map_A = self.atmospheric_light(imgPatch)*2-1
         
        trans_raw = 1 - self.omega*self.get_dark_channel(imgPatch/((map_A + 1)/2))

        T_DCP = self.guided_filter(guidance, trans_raw)
        J_DCP = (imgPatch - map_A)/T_DCP.repeat(1,3,1,1) + map_A

        return J_DCP, T_DCP, map_A
        
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
def define_D(input_nc, ndf,  init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator
    """
    
    net = None
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

    net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)

    return init_net(net, init_type, init_gain, gpu_ids)

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
    
class upsample(nn.Module):
    def __init__(self,input_nc,output_nc,kernel_size=5,use_relu=True,use_conv=True,use_norm=True, norm='IN', use_bias=False, pool_size=0) -> None:
        super(upsample,self).__init__()
        padding_size = 1

        model = []
        if use_relu:
            model += [nn.ReLU()]
        if pool_size>0:
            model += [nn.AvgPool2d(kernel_size=pool_size,stride=1,padding=pool_size//2)]
        if use_conv:
            model += [nn.ConvTranspose2d(input_nc,output_nc,kernel_size=kernel_size,stride=2,padding=padding_size,bias=use_bias)]     
        if use_norm :
            if norm == 'BN':
                  model += [nn.BatchNorm2d(output_nc)]
            elif norm == 'IN' :
                model += [nn.InstanceNorm2d(output_nc,affine=False)]
            else:
                raise NotImplementedError('normalization layer [%s] is not found' % norm)
        
        self.model = nn.Sequential(*model)
    def forward(self,x):
        x= self.model(x)
        return x 

class last_block(nn.Module):
    def __init__(self,input_nc,output_nc=1) -> None:
        super(last_block,self).__init__()

        model = []
        model += [nn.ReLU()]    
        # model += [nn.MaxPool2d(kernel_size=3,stride=1,padding=1)]
        model += [nn.Conv2d(input_nc,output_nc,kernel_size=7, stride=1, padding=3, bias = False)]     
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
   
    def forward(self,x):
        x= (self.model(x)+1)/2
        return x 

class Vgg19(torch.nn.Module):
        def __init__(self, requires_grad=False,layers=[5,12,19,25,32,38]):
            super(Vgg19, self).__init__()
            
            self.layer_id = layers
            pretrained_model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
            
            pretrained_model.eval()
            self.pretrained_model = pretrained_model.features
            if not requires_grad:
                for param in self.pretrained_model.parameters():
                    param.requires_grad = False

        def forward(self, input):
            feature = input
            output = []
            for layer_id, layer_model in enumerate(self.pretrained_model):
                # feature_saved = feature
                feature = layer_model(feature)
                if layer_id in self.layer_id:
                    output.append(feature)
                if layer_id == self.layer_id[-1]:
                    return output
            return output
            
class Transmission_estimator(torch.nn.Module): 
    def __init__(self,layers =[6, 13, 26, 39,52],output_nc = 1):
        super(Transmission_estimator,self).__init__()
        # dimList = [64, 128, 256, 512]
        self.encoder = Vgg19(layers=layers)
        self.up_1 = upsample(input_nc=512,output_nc=256)
        self.up_2 = upsample(input_nc=512+256,output_nc=256)
        self.up_3 = upsample(input_nc=256+256,output_nc=256)
        self.up_4 = upsample(input_nc=256+128,output_nc=128)
        self.up_5 = upsample(input_nc=128+64,output_nc=64)
        self.block = last_block(input_nc=64, output_nc = output_nc)
        self.transform = transforms.Compose([ transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

    def forward(self,x):
        x = self.transform(x/2+0.5)
        features = self.encoder(x)

        result_1 = self.up_1(features[4])
        result_1 = F.interpolate(result_1, size=features[3].shape[2:4], mode='nearest')

        result_2 = self.up_2(torch.cat((features[3],result_1),dim=1))
        result_2 = F.interpolate(result_2, size=features[2].shape[2:4], mode='nearest')

        result_3 = self.up_3(torch.cat((features[2],result_2),dim=1))
        result_3 = F.interpolate(result_3, size=features[1].shape[2:4], mode='nearest')

        result_4 = self.up_4(torch.cat((features[1],result_3),dim=1))
        result_4 = F.interpolate(result_4, size=features[0].shape[2:4], mode='nearest')

        result_5 = self.up_5(torch.cat((features[0],result_4),dim=1))
        result_5 = F.interpolate(result_5, size=x.shape[2:4], mode='nearest')
        
        result_final = self.block(result_5)
        return result_final
    
class Dehaze_mix(nn.Module):       
    """Define a Dehaze module based on CNN and Transformer"""

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling = 2, n_mixblock=2, norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False), use_dropout=False, use_bias=True):
        super().__init__()
        
        model = [   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]
        self.conv_in = nn.Sequential(*model)

        model = []
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                        norm_layer(ngf * mult * 2),
                        nn.ReLU(True)]
        self.downblock = nn.Sequential(*model)
        
        dim = ngf*2 ** n_downsampling
        if n_mixblock >0:
            for i in range(n_mixblock):       # add ResNet blocks
                setattr(self,"mix_"+str(i+1),  MixBlock(dim, dim//4, 8)) 

        model = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=4, stride=2,
                                            padding=1, output_padding=0,
                                            bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        nn.ReLU(True)]
        self.upblock = nn.Sequential(*model)

        model = []
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.conv_out = nn.Sequential(*model)

    def forward(self, image):
        feature = self.conv_in(image)
        feature = self.downblock(feature)

        mid_index = 1
        while hasattr(self,"mix_"+str(mid_index)):
            mixblock = getattr(self,"mix_"+str(mid_index))
            feature = mixblock(feature)
            mid_index += 1

        feature = self.upblock(feature)
        feature = self.conv_out(feature)

        return feature  

class MixBlock(nn.Module):

    def __init__(self, dim, middim, heads, kernels=[3, 5, 7, 9, 11]):
        super().__init__()
        self.kernels = kernels
        for kernel in kernels:
            model = [nn.Conv2d(dim, middim, kernel_size=kernel, stride=1, padding=kernel//2, groups=heads),
                        nn.InstanceNorm2d(middim),
                        nn.ReLU(True)]
            setattr(self,"cnn_"+str(kernel),  nn.Sequential(*model))

        self.attn = SelfAttention_custom(middim = middim, dim=middim, num_heads=heads)
        
        self.weight = nn.Sequential(*[nn.Conv2d(len(kernels)*middim, dim, kernel_size=1, stride=1),
                                      nn.ReLU(True),
                                      nn.Conv2d(dim, dim, kernel_size=3, stride=1,padding=1),])

    def forward(self, feature):
        shape = feature.shape
        for kernel in self.kernels:
            model = getattr(self,"cnn_"+str(kernel))
            
            if kernel == self.kernels[0]:
                feature_cnn = model(feature).unsqueeze(-1)
            else:
                feature_kernel = model(feature).unsqueeze(-1)
                feature_cnn = torch.cat((feature_cnn, feature_kernel), dim=-1)
        
        feature_attn = self.attn(feature_cnn) + feature_cnn.permute(0,4,1,2,3)
        feature_mix = self.weight(feature_attn.reshape(shape[0], -1, shape[2],shape[3]))

        feature_output = feature_mix + feature

        return feature_output

class SelfAttention_custom(torch.nn.Module):
    def __init__(self,  middim, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.middim = middim
        self.depth = middim // num_heads
        self.dim = dim
        self.wq = torch.nn.Linear(dim, middim)
        self.wk = torch.nn.Linear(dim, middim)
        self.wv = torch.nn.Linear(dim, middim)

        self.dense = nn.Sequential(*[nn.Linear(middim, dim),
                                    nn.LayerNorm(dim),
                                    nn.ReLU(True),
                                    nn.Linear(dim, dim),])
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product(self, query, key, value, mask=None):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))

        d_k = query.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        output = torch.matmul(attention_weights, value)
        
        return output

    def forward(self, feature, mask=None):
        batch_size = feature.size(0)
        dim = feature.size(1)
        H = feature.size(2)
        W = feature.size(3)
        feature_num = feature.size(4)

        feature = feature.permute(0,2,3,4,1).contiguous()
        feature = feature.view(-1, feature_num, dim)
        q = self.split_heads(self.wq(feature), batch_size*H*W)
        k = self.split_heads(self.wk(feature), batch_size*H*W)
        v = self.split_heads(self.wv(feature), batch_size*H*W)

        scaled_attention = self.scaled_dot_product(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()

        concat_attention = scaled_attention.view(batch_size, -1, self.middim)

        output = self.dense(concat_attention)

        output = output.reshape(batch_size, H, W, feature_num, dim)
        output  = output.permute(0,3,4,1,2).contiguous()
        return output


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

def setup_seed(seed=42):
    """
    """
    torch.manual_seed(seed)  # PyTorch CPU 
    torch.cuda.manual_seed(seed)  # PyTorch CUDA 
    torch.cuda.manual_seed_all(seed)  # PyTorch 
    np.random.seed(seed)  # NumPy 
    random.seed(seed)  # Python random 
    # torch.backends.cudnn.deterministic = True  
    # torch.backends.cudnn.benchmark = False 
    
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def synthesize_fog(J, t, A=None):
    """
    Synthesize hazy image base on optical model
    I = J * t + A * (1 - t)
    """

    if A is None:
        A = 1

    return J * t + A * (1 - t)

def dehaze_fog(I, t, A=None):
    """
    dehaze base on optical model
    I = J * t + A * (1 - t)
    """
    if A is None:
        A = 1
    return torch.div(I-A,t)+A


def is_image_file(filename):
    IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
         

class Unpaired_Dataset(Dataset):

    def __init__(self, transform, haze_dir, clear_dir):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.transform =transform
  
        self.dir_I = haze_dir 
        self.dir_J = clear_dir


        self.I_paths = sorted(make_dataset(self.dir_I))   # load images from '/path/to/data/trainA'
        self.J_paths = sorted(make_dataset(self.dir_J))    # load images from '/path/to/data/trainB'
        self.I_size = len(self.I_paths)  # get the size of dataset A
        self.J_size = len(self.J_paths)  # get the size of dataset B


    def __getitem__(self, index):
   
        I_path = self.I_paths[index % self.I_size]  # make sure index is within then range
        index_J = random.randint(0, self.J_size - 1)
        J_path = self.J_paths[index_J]

        I_img = Image.open(I_path).convert('RGB')
        J_img = Image.open(J_path).convert('RGB')

        real_I = self.transform(I_img)
        real_J = self.transform(J_img)

        return real_I,real_J
    def __len__(self):
   
        return max(self.I_size, self.J_size)

class EvalDataset(Dataset):
    def __init__(self, dataset_path_list, transform=None):
        """
        dataset_type: ['train', 'test']
        """
        self.transform = transform
        self.dir_I = os.path.join(dataset_path_list) 
        self.I_paths = sorted(make_dataset(self.dir_I))
        self.I_size = len(self.I_paths) 
        
    def __getitem__(self, index):
        I_path = self.I_paths[index % self.I_size]
        I_img = Image.open(I_path).convert('RGB')
        image_name = I_path.split('/')[-1]
        
        real_I = self.transform(I_img) # type: ignore
        return real_I, image_name

    def __len__(self):
        return self.I_size