import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
from torchvision.transforms import ToTensor

class DatasetFromFolder(data.Dataset):
    def __init__(self, imageDir):
        super(DatasetFromFolder, self).__init__()
        self.imageDir = imageDir
        self.transformer = ToTensor()

        self.guestureTypes = ['A', 'B', 'C', 'F', 'P', 'V']
        self.allSamples = []
        for g in range(0, len(self.guestureTypes)):
            for i in range(0, 400):
                sample = {}
                sample['value'] = g
                sample['file'] = join(imageDir, '{}_{}.png'.format(self.guestureTypes[g], i+1) )
                self.allSamples.append(sample)

        randp = torch.randperm(len(self.allSamples))
        for i in range(1, len(self.allSamples)):
            temp = self.allSamples[0]
            self.allSamples[0] = self.allSamples[randp[i]]
            self.allSamples[randp[i]] = temp

    def __getitem__(self, index):
        targetValue = self.allSamples[index]['value']
        imageFilePath = self.allSamples[index]['file']
        img = Image.open(imageFilePath).convert("RGB")
        img = self.transformer(img)
        img = ( img - 0.5 ) * 256
        return img, targetValue

    def __len__(self):
        return len(self.allSamples)

if __name__ == '__main__':
    d = DatasetFromFolder('./guesture')
    print( len(d) )
    print( d[0] )
