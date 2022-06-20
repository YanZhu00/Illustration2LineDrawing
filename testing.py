import torch
import os
import fnmatch
import cv2
from model import*


def loadImages(folder):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))

    return matches


model = res_skip()
model.load_state_dict(torch.load('model.pth'))
is_cuda = torch.cuda.is_available()
if is_cuda:
    model.cuda()
    print("using cuda")
else:
    model.cpu()
    print("using cuda")
model.eval()

test_folder = 'testing_inputs'
test_result = 'testing_outputs'

filelists = loadImages(test_folder)

with torch.no_grad():
    for imname in filelists:
        print("processing: ", imname)
        src = cv2.imread(imname, cv2.IMREAD_UNCHANGED)

        HSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

        rows = int(np.ceil(src.shape[0] / 16)) * 16
        cols = int(np.ceil(src.shape[1] / 16)) * 16

        # manually construct a batch. You can change it based on your usecases.
        patch = np.ones((1, 3, rows, cols), dtype="float32")
        for i in range(3):
            patch[0, i, 0:src.shape[0], 0:src.shape[1]] = HSV[0:src.shape[0], 0:src.shape[1], i]

        patch[0][0] = patch[0][0] / 180
        patch[0][1] = patch[0][1] / 255
        patch[0][2] = patch[0][2] / 255

        if is_cuda:
            tensor = torch.from_numpy(patch).cuda()
        else:
            tensor = torch.from_numpy(patch).cpu()

        y = model(tensor)

        yc = y.cpu().numpy()[0, 0, :, :]
        yc[yc > 1] = 1
        yc[yc < 0] = 0
        yc = yc*255

        head, tail = os.path.split(imname)
        cv2.imwrite(test_result+"/"+tail.replace(".jpg", ".png"), yc[0:src.shape[0], 0:src.shape[1]])