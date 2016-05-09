local flags = {}

local classes = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"}

flags.allDB = "./boxDB.json"
flags.imagePath = "/home/teaonly/dataset/voc/tools/images/"
--flags.imagePath = "/Users/teaonly/Workspace/ml/voc/images"

flags.classmap = {}
for i = 1, #classes do
    flags.classmap[i] = classes[i]
    flags.classmap[classes[i]] = i
end

flags.imageWidth = 448
flags.imageHeight = 448
flags.grid = 8


return flags
