from PIL import Image 
# image_file = Image.open("convert_image.png") # open colour image
# image_file = image_file.convert('1') # convert image to black and white
# image_file.save('result.png')

from os import listdir, getcwd, makedirs
from os.path import isfile, join, exists

datasetPath = "D:\\UMN\\SKRIPSI\\datasets\\Training - Cleaned"
datasetFolders = ["Ceplok", "Kawung", "Lereng", "Nitik", "Parang"]

currDir = getcwd()
savePath = "bwDatasets"

imagesPath = []
for a in datasetFolders:
  fld = join(datasetPath, a)
  for f in listdir(fld):
    imgFile = join(fld,f)
    img = Image.open(imgFile)
    img = img.convert('1')
    newFld = a+"_bw"

    if not exists(newFld):
      makedirs(newFld)
    
    img.save(join(currDir,newFld)+"\\"+f)

