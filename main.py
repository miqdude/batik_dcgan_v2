from torch import nn
import torch
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import torchvision.utils as vutils
import torch.nn.parallel
import torch.optim as optim


# variables
DATASETFOLDER = ["Ceplok", "Kawung", "Lereng", "Nitik", "Parang"]

BATCH_SIZE = 128
DATASET_ROOT_DIR = r'F:\miqdude\Batik'
# 'C:\Users\CORE\Desktop\miqdude\Kawung'
LEARNING_RATE = 0.002
EPOCH = 10000
SAVE_INTERVAL = 100
NOISE_VECTOR_DIM = 100
FEATURE_MAPS = 64
CHANNEL_SIZE = 3        # image channel RGB

# check CUDA Support GPU
cuda = True if torch.cuda.is_available() else False
print("cuda availability {}" % cuda)

# initiate weight at class declaration
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##
# Generator object 
# ##
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        """ 
        Inverse transpose convolution network
        tuned from DCGAN paper
        """
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NOISE_VECTOR_DIM, FEATURE_MAPS * 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(FEATURE_MAPS * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURE_MAPS * 8, FEATURE_MAPS * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(FEATURE_MAPS * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURE_MAPS * 4, FEATURE_MAPS * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(FEATURE_MAPS * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURE_MAPS * 2, FEATURE_MAPS, 4, 2, 1, bias = False),
            nn.BatchNorm2d(FEATURE_MAPS),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURE_MAPS, CHANNEL_SIZE, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    # function called inside __init__
    def forward(self, input):
        output = self.main(input)
        return output

##
# Discriminator object
# ##
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        """ 
        convolution network tuned   
        according to paper DCGAN by Alec Radford
        """
        self.main = nn.Sequential(
            nn.Conv2d(CHANNEL_SIZE, FEATURE_MAPS, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(FEATURE_MAPS, FEATURE_MAPS * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(FEATURE_MAPS * 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(FEATURE_MAPS * 2, FEATURE_MAPS * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(FEATURE_MAPS * 4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(FEATURE_MAPS * 4, FEATURE_MAPS * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(FEATURE_MAPS * 8),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(FEATURE_MAPS * 8, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)
        

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# enable cuda compute on generator, discriminator, and loss function
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader for feeding data to discriminator every batch
dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        DATASET_ROOT_DIR,
        transform = transforms.Compose(
            [transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# # Optimizers using Adam
# optimizerG = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
# optimizerD = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Optimizers using RMSProp
optimizerG = torch.optim.SGD(generator.parameters(), lr=LEARNING_RATE, momentum =0.5)
optimizerD = torch.optim.SGD(discriminator.parameters(), lr=LEARNING_RATE, momentum=0.5)


# Tensor object for gpu compute
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print("Training Start.....")

# The training process
for epoch in range(EPOCH):

    for i, data in enumerate(dataloader, 0):
        
        # Train discriminator with real images
        discriminator.zero_grad() # clears the gradient
        
        real, _ = data
        input = Variable(real.type(Tensor))  # input images must be Tensor for gpu compute
        target = Variable(torch.ones(input.size()[0]).cuda())
        output = discriminator(input)
        errD_real = adversarial_loss(output, target)
        

        # Train discriminator with fake images
        # Generator generate fake images
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1).cuda())
        fake = generator(noise)
        target = Variable(torch.zeros(input.size()[0]).cuda())
        
    
        output = discriminator(fake.detach()).view(-1)
        errD_fake = adversarial_loss(output, target)

        # calculate error/loss on discriminator    
        errD = errD_real + errD_fake
        errD.backward() # backward propagation
        optimizerD.step() # optimizing model

        # Train generator
        generator.zero_grad()
        target = Variable(torch.ones(input.size()[0]).cuda())
        output = discriminator(fake)
        errG = adversarial_loss(output, target)
        errG.backward()
        optimizerG.step()
        
        # print training logs
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, EPOCH, i, len(dataloader), errD.data, errG.data))
        
        # logging training loss
        with open("training_log.txt", "a") as myfile:
            myfile.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f\n' % (epoch, EPOCH, i, len(dataloader), errD.data, errG.data))
            myfile.close()

        # Save images and models per interval and last epoch
        if (epoch % SAVE_INTERVAL) == 0 or (epoch == EPOCH - 1):
            vutils.save_image(real, '%s/real_samples_%03d.png' % ("./results",epoch), normalize = True)
            fake = generator(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)

        # save models once very 1000th epoch and last epoch
        if (epoch % 2000 == 0) or (epoch == EPOCH -1):  
            # save models for future use
            # save generator 
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizerG.state_dict(),
                'loss': errG,
            }, './models/generator_%03d.pt' % epoch)

            # save discriminator 
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict(),
                'loss': errD,
            }, './models/discriminator_%03d.pt' % epoch)