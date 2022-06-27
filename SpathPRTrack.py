# -*- coding: utf-8 -*
fromloguru importlogger
fromtorch.autograd importVariable
importtorch
importtorch.nn as nn
from videoanalyst.model.common_opr.common_block import(conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import(TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)
torch.set_printoptions(precision=8)
class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = C
        # Output_dim = C (number of channels)
        self.To_latent =nn.Sequential(
            #State (cx64x1) {Cx[data_size]}
            nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=4, stride=2,padding=1),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (64x32x1)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (128x16x1)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
 
            # State (256x8x1)
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State(512x4x1)
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=3, padding=0))
            
        self.To_syn =nn.Sequential(
            # Self.To_latent's output[2]
            # State (1x1x1)
            nn.ConvTranspose1d(in_channels=channels, out_channels=512, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(True),
            # State (512x4x1)
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(True),
            # State (256x8x1)
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(True),
            # State (128x16x1)
            nn.ConvTranspose1d(in_channels=128, out_channels=channels, kernel_size=4, stride=2, padding=1)
)
            # output of To_real -> #State(1x32x1)
        self.output =nn.Tanh()
    def forward(self, x):
        middle =x.size()[1]//2
        x =x.unsqueeze(1)
        front =x[:,:,:middle]
        z =self.To_latent(x)
        z =z[:,:,1].unsqueeze(1)
        z =self.To_syn(z)
        z =self.output(z)
        return torch.cat([front,z],dim=2)
    
class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [128, 256, 512]
        # Input_dim = channels (batchx1x64)
        # Output_dim = 1
        self.main_module =nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Series (1x64x1)
            nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (128x32x1)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (256x16x1)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        
            # State (512x8x1)
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            # State (512x4x1)
            # output of main module --> State
)
        self.output =nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=4, padding=0))

    def forward(self, x):
        x=self.main_module(x)
        return self.output(x)
    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x =self.main_module(x)
        return x.view(-1, 512*4*1)
@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register

class STMTrack(ModuleBase):
    default_hyper_params =dict(pretrain_model_path="",
                                head_width=256,
                                conv_weight_std=0.01,
                                corr_fea_output=False,
                                amp=False)
    support_phases =["train", "memorize", "track"]
    def __init__(self, backbone_m, backbone_q, neck_m, neck_q, head, loss=None):
        super(STMTrack, self).__init__()
        self.basemodel_m =backbone_m
        self.basemodel_q =backbone_q
        self.neck_m =neck_m
        self.neck_q =neck_q
        self.head =head
        self.loss =loss
        self._phase ="train"
    @property
    def phase(self):
        return self._phase
    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase =p
    def memorize(self, im_crop, fg_bg_label_map):
        fm =self.basemodel_m(im_crop, fg_bg_label_map)
        fm =self.neck_m(fm)
        fm =fm.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # B, C, T, H, W
        return fm
    def train_forward(self, training_data):
        memory_img =training_data["im_m"]
        query_img =training_data["im_q"]
        # backbone feature
        assert len(memory_img.shape) ==5
        B, T, C, H, W =memory_img.shape
        memory_img =memory_img.view(-1, C, H, W)  # no memory copy
        target_fg_bg_label_map =training_data["fg_bg_label_map"].view(-1, 1, H, W)
        fm =self.basemodel_m(memory_img, target_fg_bg_label_map)
        fm =self.neck_m(fm)  # B * T, C, H, W
        fm =fm.view(B, T, *fm.shape[-3:]).contiguous()  # B, T, C, H, W
        fm =fm.permute(0, 2, 1, 3, 4).contiguous()  # B, C, T, H, W
        fq =self.basemodel_q(query_img)
        fq =self.neck_q(fq)
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea =self.head(fm, fq)
        predict_data =dict(
            cls_pred=fcos_cls_score_final,
            ctr_pred=fcos_ctr_score_final,
            box_pred=fcos_bbox_final,
)
        if self._hyper_params["corr_fea_output"]:
            predict_data["corr_fea"] =corr_fea
        return predict_data
    def forward(self, *args, phase=None):
        if phase is None:
            phase =self._phase
        # used during training
        if phase =='train':
            # resolve training data
            if self._hyper_params["amp"]:
                with torch.cuda.amp.autocast():
                    return self.train_forward(args[0])
            else:
                return self.train_forward(args[0])
        elif phase =='memorize':
            target_img, fg_bg_label_map =args
            fm =self.memorize(target_img, fg_bg_label_map)
            out_list =fm
        elif phase =='track':
            assert len(args) ==2
            search_img, fm =args
            fq =self.basemodel_q(search_img)
            fq =self.neck_q(fq)  # B, C, H, W
            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea =self.head(
                fm, fq, search_img.size(-1))
            # apply sigmoid
            fcos_cls_prob_final =torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final =torch.sigmoid(fcos_ctr_score_final)
            # apply centerness correction
            fcos_score_final =fcos_cls_prob_final *fcos_ctr_prob_final
            extra =dict()
            # output
            out_list =fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra
        else:
            raise ValueError("Phase non-implemented.")
        return out_list
    def update_params(self):
        self._make_convs()
        self._initialize_conv()
        super().update_params()
    def _make_convs(self):
        head_width =self._hyper_params['head_width']
        # feature adjustment
        self.r_z_k =conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_z_k =conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.r_x =conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_x =conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
    def _initialize_conv(self, ):
        conv_weight_std =self._hyper_params['conv_weight_std']
        conv_list =[
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
]
        for ith in range(len(conv_list)):
            conv =conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01
    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev =torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
    def memoooo(self, dev):
        Tensor =torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # Loss weight for gradient penalty
        lambda_gp =10
        # Initialize generator and discriminator
        generator =Generator(1)
        discriminator =Discriminator(1)
        data_shape =(1, 64) #(batch_size, 1, 64)
        cuda =True if torch.cuda.is_available() else False
        lr =1e-3
        b1 =0.5
        b2 =0.999
        batch_size =256
        n_epochs =200
        n_critic =5 #number of training steps for discriminator per iter\
        # Optimizers
        optimizer_G =torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D =torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

        for i, data in enumerate(memory_block):
        # Configure input
        #real_imgs = Variable(data['data'].type(Tensor))
        syn_series =Variable(data['syn_data'].type(Tensor))
        real_series =Variable(data['data'].type(Tensor))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Sample noise as generator input
        fake_series =generator(syn_series).squeeze(1)
        
        # Real
        real_validity =discriminator(real_series.unsqueeze(1))
        # Fake
        fake_validity =discriminator(fake_series.unsqueeze(1))
        # Gradient penalty
        gradient_penalty =compute_gradient_penalty(discriminator, real_series, fake_series)
        L2distance =F.pairwise_distance(real_series[:,32:],fake_series[:,32:])#.view(-1,1,1)
        # Adversarial loss
        d_loss =-torch.mean(real_validity) +torch.mean(fake_validity) +lambda_gp *gradient_penalty
    
        d_loss.backward(retain_graph=True)
        optimizer_D.step()
        optimizer_G.zero_grad()
        # Train the generator every n_critic steps
        if i %n_critic ==0:
            # -----------------
            #  Train Generator
            # -----------------
            # Generate a batch of images
            fake_series =generator(syn_series).squeeze(1)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity =discriminator(fake_series.unsqueeze(1))
           
            L2distance =F.pairwise_distance(real_series[:,32:],fake_series[:,32:])#.view(-1,1,1)
      
            point_loss =torch.mean(Variable(L2distance).type(torch.float32))
            
            g_loss =-torch.mean(fake_validity)-point_loss*0.05
            g_loss.backward()
            optimizer_G.step()
            
epoch_Gloss+=abs(g_loss.item())
            
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                %(epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
)
            ifepoch>n_epochs-5:
                torch.save(generator.state_dict(),'summary/testing_model/generator'+str(epoch)+'.pth')
        

최종 결과물 4. 제안하는 새로운 Box Regression기법-코드구현

importpymagsac
importcv2
import numpy as np
def detect_box(feature_map, scale):
    #make box directly from feature map
    Essential_Line_x =pymagsac.prosac_sampler(feature_map[x])
    Essential_Line_y =pymagsac.prosac_sampler(feature_map[y])
    return scale*[Essential_Line_x,Essential_Line_y]
def xywh2xyxy(rect):
        rect =np.array(rect, dtype=np.float32)
        return np.concatenate([
            rect[..., [0]], rect[..., [1]], rect[..., [2]] +rect[..., [0]] -1,
            rect[..., [3]] +rect[..., [1]] -1
],axis=-1)
def createFolder(Essential_Line,img): 
    bbox_pred =tuple(map(float,Essential_Line.split(',')))
    bbox_pred =xywh2xyxy(bbox_pred)
    bbox_pred =tuple(map(int, bbox_pred))
    cv2.rectangle(img, bbox_pred[:2], bbox_pred[2:],(0, 255, 0))
    height, width, layers =img.shape
    size =(width,height)   
    #Scaler