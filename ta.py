import torch
import argparse
import networks.D2E as net
import torchvision
import tensorboardX
import os
import numpy as np
import pytorch_ssim

use_gpu = True
device = torch.device("cuda" if use_gpu else "cpu")
#G = net.Generator(input_dim=256, output_channels = 3, image_size=256, Gscale=8, another_times=0).to(device)
#G.load_state_dict(torch.load('./preTrained_model/Epoch_G_99.pth',map_location=device)) #shadow的效果要好一些 
G = net.Generator(input_dim=128, output_channels = 3, image_size=128, Gscale=16, another_times=0).to(device)
G.load_state_dict(torch.load('../preTrained_model/Img128_Gs16_Ds1_Zdim128.pth',map_location=device)) #shadow的效果要好一些 
G.eval() #针对batch_norm的采样

E = net.Discriminator_SpectrualNorm(input_dim=128, input_channels = 3, image_size=128, Gscale=16, Dscale=1, another_times=0).to(device)
print(E)

E_optimizer = torch.optim.Adam(E.parameters(), lr=0.0002, betas=(0.0, 0.999), eps=1e-8)
mse_loss = torch.nn.MSELoss()
ssim_loss = pytorch_ssim.SSIM()

def set_seed(seed): #随机数设置
    np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


output_dir = os.path.join('./output/', 'EAE_128_ssim')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

ckpt_dir = os.path.join(output_dir, 'checkpoints')
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

sample_dir = os.path.join(output_dir, 'samples_training')
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

writer = tensorboardX.SummaryWriter(os.path.join(output_dir, 'summaries'))

it_d=0
batch_size = 100
for epoch in range(0,250001):
        set_seed(epoch%30000)
        z1 = torch.randn(batch_size, 128, 1, 1).cuda() 
        with torch.no_grad(): 
            imgs1 = G(z1).cuda()
        z2 = E(imgs1)
        imgs2=G(z2)

        loss_img = ssim_loss(imgs1,imgs2)
        loss_w = ssim_loss(z1,z2)

        loss = loss_mse_img+loss_mse_z
        E_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        E_optimizer.step()

        print('i_'+str(epoch))
        print('---------ImageSpace--------')
        print('loss_img:'+str(loss_img.item()))
        print('---------LatentSpace--------')
        print('loss_w:'+str(loss_w.item()))

        it_d += 1
        writer.add_scalar('loss_img' , loss_img.detach().cpu().numpy(), global_step=it_d)
        writer.add_scalar('loss_w' , loss_w.detach().cpu().numpy(), global_step=it_d)
        writer.add_scalars('Double_Space', {'loss_mse_img_':loss_img,'loss_w_':loss_w}, global_step=it_d)

        if epoch % 100 == 0:
            n_row = 40
            test_img = torch.cat((imgs1[:n_row],imgs2[:n_row])) #*0.5+0.5
            torchvision.utils.save_image(test_img, sample_dir+'/ep%d.jpg'%(epoch),nrow=n_row//2) # nrow=3
            with open(output_dir+'/Loss.txt', 'a+') as f:
                        print('i_'+str(epoch),f)
                        print('---------ImageSpace--------',f)
                        print('loss_mse_img:'+str(loss_mse_img.item()),f)
                        print('---------LatentSpace--------',f)
                        print('loss_w:'+str(loss_mse_z.item()),f)
            if epoch % 5000 == 0:
                torch.save(E.state_dict(), ckpt_dir+'/E_model_ep%d.pth'%epoch)
                #torch.save(Gm.buffer1,resultPath1_2+'/center_tensor_ep%d.pt'%epoch)



