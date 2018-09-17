import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

input_height, input_width = 96, 96
## output shape is the same as input
output_height, output_width = input_height, input_width
n = 32*5
nClasses = 15
nfmp_block1 = 64
nfmp_block2 = 128
nb_epochs = 300
batch_size = 32
const = 10

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, nfmp_block1, 3, padding=1)
        self.act1 = F.relu
        self.conv2 = nn.Conv2d(nfmp_block1, nfmp_block1, 3, padding=1)
        self.act2 = F.relu
        self.block1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(nfmp_block1, nfmp_block2, 3, padding=1)
        self.act3 = F.relu
        self.conv4 = nn.Conv2d(nfmp_block2, nfmp_block2, 3, padding=1)
        self.act4 = F.relu
        self.block2 = nn.MaxPool2d(2)

        ## bottoleneck
        # self.bl_conv1 = nn.Conv2d(nfmp_block2, n, (input_height//4, input_width//4),
        #     padding='same') 
        '''
        doing padding = same if odd sized kernel
        '''
        cur_h = input_height//4
        cur_w = input_width//4
        pad_h = cur_h//2
        pad_w = cur_w//2
        if cur_h % 2 == 0:
            pad_h1 = pad_h
            pad_h2 = pad_h - 1
        else:
            pad_h1 = pad_h2 = pad_h
        if cur_w % 2 == 0:
            pad_w1 = pad_w
            pad_w2 = pad_w - 1
        else:
            pad_w1 = pad_w2 = pad_w
        self.bl_pad = nn.ZeroPad2d((pad_h1, pad_h2, pad_w1, pad_w2))
        self.bl_conv1 = nn.Conv2d(nfmp_block2, n, (cur_h, cur_w), padding=0)
        self.bl_conv2 = nn.Conv2d(n, n, 1, padding=0)
        self.relu = F.relu
        '''
        upsample
        '''
        self.upsample = nn.ConvTranspose2d(n, nClasses, (4, 4), stride=(4, 4))
        
    def forward(self, img):
        x = self.conv1(img)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.block1(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.block2(x)
        x = self.relu(self.bl_conv1(self.bl_pad(x)))
        x = self.relu(self.bl_conv2(x))
        x = self.upsample(x)
        x = x.view(-1, output_width * output_height * nClasses, 1)
        return x


class test(unittest.TestCase):
    def test1(self):
        batch = torch.randn(64, 1, 96, 96).cuda()
        model = Encoder().cuda()
        output = model(batch)
        print(output.shape)

    def test2(self):
        pass

if __name__ == "__main__":
    unittest.main()
