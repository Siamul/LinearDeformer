import math
import torch
from torch.nn.functional import grid_sample
import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

class ResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv_res = nn.Conv2d(in_channels, out_channels, 1, stride =  1, bias = False)
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class NestedResUNetParam(nn.Module):
    def __init__(self, num_channels, num_params=6, width=32, resolution=(240, 320)):
        super().__init__()

        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = nn.MaxPool2d(2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv0_0 = ResBlock(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = ResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = ResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = ResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = ResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = ResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = ResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        
        self.xyr_input = ResBlock(num_channels, nb_filter[0], nb_filter[0])      
        self.xyr_conv1 = ResBlock(nb_filter[0]*5, nb_filter[0]*4, nb_filter[0]*4)
        self.xyr_conv2 = ResBlock(nb_filter[0]*4, nb_filter[0]*3, nb_filter[0]*3)
        self.xyr_conv3 = ResBlock(nb_filter[0]*3, nb_filter[0]*2, nb_filter[0]*2)
        self.xyr_conv4 = ResBlock(nb_filter[0]*2, nb_filter[0], nb_filter[0])
        self.xyr_linear = nn.Sequential(
                              nn.Flatten(),
                              nn.Linear(nb_filter[0]*int(resolution[0]/16)*int(resolution[1]/16), int(resolution[0]/16)*int(resolution[1]/16)),
                              nn.ReLU(),
                              nn.Linear(int(resolution[0]/16)*int(resolution[1]/16), num_params)
                          )


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        xyr0 = self.xyr_input(input)
        xyr1 = self.xyr_conv1(torch.cat([xyr0, x0_1, x0_2, x0_3, x0_4], 1)) #240x320
        xyr2 = self.xyr_conv2(self.pool(xyr1)) #120x160
        xyr3 = self.xyr_conv3(self.pool(xyr2)) #60x80
        xyr4 = self.xyr_conv4(self.pool(xyr3)) #30x40
        xyr5 = self.xyr_linear(self.pool(xyr4)) #15x20
          
        return xyr5



class LinearDeformer(object):
    def __init__(self, net_path = './nestedresunetparam-best.pth', device=torch.device('cuda')):
        self.net_path = net_path
        self.device = device
        self.NET_INPUT_SIZE = (320,240)
        self.model = NestedResUNetParam(1, 6)
        try:
            self.model.load_state_dict(torch.load(self.net_path, map_location=self.device))
        except AssertionError:
                print("assertion error")
                self.model.load_state_dict(torch.load(self.net_path,
                    map_location = lambda storage, loc: storage))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.input_transform = Compose([
            ToTensor(),
            Normalize(mean=(0.5791223733793273,), std=(0.21176097694558188,))
        ])
        
    def circApprox(self, image):
        w,h = image.size
        image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
        w_mult = w/self.NET_INPUT_SIZE[0]
        h_mult = h/self.NET_INPUT_SIZE[1]

        inp_xyr_t = self.model(Variable(self.input_transform(image).unsqueeze(0).to(self.device)))

        #Circle params
        inp_xyr = inp_xyr_t.tolist()[0]
        pupil_x = int(inp_xyr[0] * w_mult)
        pupil_y = int(inp_xyr[1] * h_mult)
        pupil_r = int(inp_xyr[2] * max(w_mult, h_mult))
        iris_x = int(inp_xyr[3] * w_mult)
        iris_y = int(inp_xyr[4] * h_mult)
        iris_r = int(inp_xyr[5] * max(w_mult, h_mult))

        return np.array([pupil_x,pupil_y,pupil_r]), np.array([iris_x,iris_y,iris_r])
    
    
    def find_xy_between(self, r1, r2, r1p, r3p, xp, yp, xc, yc):
        r3 = (((r3p - r1p) * (r2 - r1)) / (r2 - r1p)) + r1
        x = ((r3 * (xp - xc)) / r3p) + xc
        y = ((r3 * (yp - yc)) / r3p) + yc
        return x, y
    
    def find_xy_less(self, r1, r1p, xp, yp, xc, yc):
        x = ((r1 * (xp - xc)) / r1p) + xc
        y = ((r1 * (yp - yc)) / r1p) + yc
        return x, y
    '''
    def create_grid(self, r1, r2, r1p, h, w, xc, yc):
        grid = torch.zeros((h, w, 2))
        for xp in range(w):
            for yp in range(h):
                #r3p = math.dist((xp, yp), (xc, yc))
                #if r3p == 0 or r3p >= r2:
                grid[yp, xp, 0] = xp
                grid[yp, xp, 1] = yp
                #elif r3p > r1p and r3p < r2:
                #    x, y = self.find_xy_between(r1, r2, r1p, r3p, xp, yp, xc, yc)
                #    grid[yp, xp, 0] = x
                #    grid[yp, xp, 1] = y
                #else:
                #    x, y = self.find_xy_less(r1, r1p, xp, yp, xc, yc)
                #    grid[yp, xp, 0] = x
                #    grid[yp, xp, 1] = y
       #grid = grid.unsqueeze(0)
       # print(grid)

        Y = torch.arange(0, h).reshape(1, h, 1).repeat(1, 1, w).float()
        X = torch.arange(0, w).reshape(1, 1, w).repeat(1, h, 1).float()

        no_change = torch.cat((X.unsqueeze(3), Y.unsqueeze(3)), dim=3)

        #print(torch.sum(grid - no_change))

        return no_change
    '''
    def create_grid(self, r1, r2, r1p, b, h, w, xc, yc): # batched implementation
        Y = torch.arange(0, h).reshape(1, h, 1).repeat(b, 1, w).float()
        X = torch.arange(0, w).reshape(1, 1, w).repeat(b, h, 1).float()

        r3p = torch.sqrt((X - xc) ** 2 + (Y - yc) ** 2) #(b,h,w)
        r1r = r1.reshape(-1, 1, 1).repeat(1, h, w)
        r2r = r2.reshape(-1, 1, 1).repeat(1, h, w)
        r1pr = r1p.reshape(-1, 1, 1).repeat(1, h, w)
        xcr = xc.reshape(-1, 1, 1).repeat(1, h, w)
        ycr = yc.reshape(-1, 1, 1).repeat(1, h, w)

        no_change_cond = torch.where(r3p >= r2r, 1, 0).unsqueeze(3).repeat(1,1,1,2)
        no_change = torch.cat((X.unsqueeze(3), Y.unsqueeze(3)), dim=3)
        xy_between_cond = torch.where(torch.logical_and(r3p > r1pr, r3p < r2r), 1, 0).unsqueeze(3).repeat(1,1,1,2)
        x_between, y_between = self.find_xy_between(r1r, r2r, r1pr, r3p, X, Y, xcr, ycr)
        xy_between = torch.cat((x_between.unsqueeze(3), y_between.unsqueeze(3)), dim=3)

        xy_less_cond = torch.where(r3p <= r1pr, 1, 0).unsqueeze(3).repeat(1,1,1,2)
        x_less, y_less = self.find_xy_less(r1r, r1pr, X, Y, xcr, ycr)
        xy_less = torch.cat((x_less.unsqueeze(3), y_less.unsqueeze(3)), dim=3)

        grid = (no_change_cond * no_change + xy_between_cond * xy_between + xy_less_cond * xy_less).float()

        return grid, np.uint8((r3p > r1pr).numpy()[0])
    
    def deform(self, input, grid, interp_mode):  #helper function
    
        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = gridx / (W - 1)
        gridx = (gridx - 0.5) * 2
        gridy = gridy / (H - 1)
        gridy = (gridy - 0.5) * 2
        #gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        #gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1)
        return grid_sample(input, newgrid, mode=interp_mode, align_corners=True)
    
    def get_alpha(self, image):
        pupil_xyr, iris_xyr = self.circApprox(image)
        return pupil_xyr[2]/iris_xyr[2]
    
    def linear_deform(self, image, alpha):
        pupil_xyr, iris_xyr = self.circApprox(image)
        xc = torch.tensor((pupil_xyr[0] + iris_xyr[0])/2).unsqueeze(0).float()
        yc = torch.tensor((pupil_xyr[1] + iris_xyr[1])/2).unsqueeze(0).float()
        r1 = torch.tensor(pupil_xyr[2]).unsqueeze(0).float()
        r2 = torch.tensor(iris_xyr[2]).unsqueeze(0).float()
        r1p = r2 * torch.tensor(alpha).unsqueeze(0).float()
        image = ToTensor()(image).unsqueeze(0) * 255
        w = image.shape[3]
        h = image.shape[2]
        newgrid, pupil_mask = self.create_grid(r1, r2, r1p, 1, h, w, xc, yc)
        #print(newgrid)
        #torch.save(newgrid, 'newgrid.pt')
        deformed_image = self.deform(image, newgrid, interp_mode='bilinear')
        deformed_image = torch.clamp(torch.round(deformed_image), min=0, max=255)
        return Image.fromarray(deformed_image[0][0].cpu().numpy().astype(np.uint8) * pupil_mask, 'L')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_path', default='./nestedresunetparam-best.pth')
    parser.add_argument('--image_path', default='./images/iris_image6.png')
    parser.add_argument('--save_dir', default='./deformed_images/')
    parser.add_argument('--alpha', type=float, default=0.5) #The ratio of the target pupil radius with the iris radius
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    linearDeformer = LinearDeformer(net_path=args.net_path, device=device)
    image_pil = Image.fromarray(np.array(Image.open(args.image_path).convert('RGB'))[:, :, 0], 'L')
    deformed_image = linearDeformer.linear_deform(image_pil, 0.6)
    deformed_image.save(os.path.join(args.save_dir, os.path.split(args.image_path)[1]))