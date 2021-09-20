import cv2
import datetime
import numpy as np
import os
import json
import torch
from PIL import Image
from PIL import ImageDraw , ImageFont
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib


from mask_extraction import mask_extraction
from predict import predict
from other_tools import makedir, count


for a in range(126, 127):
    path = "gc{}".format(a)
    path1 = "gc{}\\xor extracted".format(a)
    path2 = "gc{}\\original".format(a)
    path3 = "gc{}\\size_100".format(a)
    path4 = "gc{}\\size_128".format(a)
    makedir(path)
    makedir(path1)
    makedir(path2)
    makedir(path3)
    makedir(path4)
    cap = cv2.VideoCapture("gc{}.mp4".format(a))
    ret, frame = cap.read()
    past = frame
    n = 1
    x = 0
    interval = 4
    print('present time', datetime.datetime.now())
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('start extract video, fps is', fps)
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    outVideo = cv2.VideoWriter('gc{}\\xor.avi'.format(a), fourcc, 50.0, (width, height))

    # extract frames from the video, operate the bitwise_xor algorithm, and save the result.
    while cap.isOpened() is True:
        ret, frame = cap.read()
        if frame is None:
            cap.release()
            cv2.destroyAllWindows()
            break
        else:
            if (n % interval) == 0:
                x = x + 1
                frame_xor = np.bitwise_xor(frame, past)
                past = frame
                cv2.imwrite("gc{}\\orig_{}.jpg".format(a, x), frame)
                cv2.imwrite("gc{}\\original\\orig_{}.jpg".format(a, x), frame)
                cv2.imwrite("gc{}\\xor_{}.jpg".format(a, x), frame_xor)
            cv2.waitKey(1)
            n = n + 1
    # '''
    print('video extraction finished')
    k = 0
    id = []
    num = count('gc{}'.format(a))
    # this step is to pick the continuous stable frames out of a video, the method is calculate the differences
    # between neighbour frames, if the difference is smaller than 30000, I will count that frame as a stable one
    for i in range(1, num - 3):
        frame = cv2.imread("gc{}\\orig_{}.jpg".format(a, i))
        frame2 = cv2.imread("gc{}\\orig_{}.jpg".format(a, i + 1))
        frame3 = cv2.absdiff(frame, frame2)
        frame4 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
        # resize the frame to reduce the calculation
        frame4.resize((192, 108), refcheck=False)
        add = 0
        for j in range(192):
            add = add + sum(frame4[j])
        print(i, add)
        if add < 30000:
            k = k + 1
            id.append(i)
    array = np.zeros((num - 3), dtype=np.uint8)

    # this section is to judge whether one stable frame is from a continuous stable video
    # and classify the video into several pieces, each of them belongs to one stable, continuous sub video,
    # which means during this sub video, the camera is not moving. So that I can analyse the whole
    # sub video and label these frames with a voted result to reduce the noise.

    # the method to judge continuity is by checking whether one frame and its two neighbours are all stable frames.
    array[0] = 1
    for i in range(1, num-3):
        if (i + 1) in id or (i + 2) in id or (i - 2) in id or i in id or (i - 1) in id:
            array[i] = 1
    array1 = np.ones((num - 3), dtype=np.uint8)
    array1[0] = 0
    for i in range(1, num-5):
        if array[i - 1] == 0 or array[i] == 0 or array[i + 1] == 0 or array[i - 2] == 0 or array[i + 2] == 0:
            array1[i] = 0
    id = []
    for i in range(1, num - 3):
        if array1[i] == 1:
            id.append(i)

    area = []
    sub_area = []
    for i in range(1, num - 3):
        if i in id:
            if (i + 1) in id and (i + 2) in id:
                key = 1
            elif (i - 1) in id and (i + 1) in id:
                key = 1
            elif (i - 1) in id and (i - 2) in id:
                key = 1
            else:
                key = 0
            if key == 1:
                sub_area.append(i)
        else:
            if len(sub_area) >= 8:
                area.append(sub_area)
                sub_area = []
            else:
                sub_area = []
    # 'area' is a list of the labels of all continuous, stable frame, and 'sub_arae' contains
    # labels of one continuous, stable frame
    area.append(sub_area)
    id1 = np.array(area)

    # next step is to extract texture information from all stable, continuous images
    information = []
    for i1 in range(len(area)):
        sub_infor = []
        for i2 in range(len(area[i1])):
            i = area[i1][i2]
            frame = cv2.imread("gc{}\\orig_{}.jpg".format(a, i))
            frame_xor = cv2.imread("gc{}\\xor_{}.jpg".format(a, i))
            img_info = []
            img_info.append(i)
            # reduce the size of the image to reduce calculation
            h, l = frame_xor.shape[:2]
            frame_xor2 = cv2.resize(frame_xor, (int(l / 3), int(h / 3)))
            # mask = mask_extraction(frame)
            k = 41  # set the kernel size of Gaussian filter in information extraction
            k_size = 3  # kernel size of dilation in mask extraction
            iterations = 5  # iteration of dilation in mask extraction

            # next step is to extract the cube's surface information from the original image
            # contains cube's surface as masks, coordinates, and the location of the centre of the mask,
            masks, location, y_cords, x_cords = mask_extraction(frame, k_size, iterations)
            # print(x_cords, y_cords)
            locations = np.array(location)
            mask_number = len(masks)
            np.array(masks)
            if mask_number > 1:
                if x_cords[0] < x_cords[1]:
                    order = [0, 1]
                else:
                    order = [1, 0]
            if mask_number == 1:
                order = [0, 0]
            img_info.append(mask_number)

            # PLT_frame = Image.fromarray(frame)
            # next step is to judge the orientation of the mask, due to the xor texture do not have rotation invariance
            # we have to define the orientation of each texture's image.
            for j in range(mask_number):
                img_pixel = []
                img_pixel.append(order[j])
                img_pixel.append(x_cords[j] * 3)
                img_pixel.append(y_cords[j] * 3)
                for ii in range(4):
                    if locations[j, ii, 0] < x_cords[j] and locations[j, ii, 1] < y_cords[j]:
                        A = ii
                    if locations[j, ii, 0] > x_cords[j] and locations[j, ii, 1] < y_cords[j]:
                        B = ii
                    if locations[j, ii, 0] > x_cords[j] and locations[j, ii, 1] > y_cords[j]:
                        C = ii
                    if locations[j, ii, 0] < x_cords[j] and locations[j, ii, 1] > y_cords[j]:
                        D = ii
                loca = np.array([[locations[j, A, 0], locations[j, A, 1]], [locations[j, B, 0], locations[j, B, 1]],
                                 [locations[j, C, 0], locations[j, C, 1]], [locations[j, D, 0], locations[j, D, 1]]])

                # combine the mask and the texture information. finally we got what we want, them reshape the result
                # into a size of 128*128 to normalize it.
                final_res = np.zeros((128, 128), dtype=np.uint8)
                trans = np.float32([[0, 0], [127, 0], [127, 127], [0, 127]])

                src = np.float32(loca)
                # use 'cv2.getPerspectiveTransform' and 'cv2.warpPerspective()' to turn the irregular
                # shape into a 128*128 sized square.
                h = cv2.getPerspectiveTransform(src, trans)
                wrapped_xor = cv2.warpPerspective(frame_xor2, h, (128, 128))
                # here we prepared two size for testing, first, 128*128, second , 100*100
                wrapped_xor2 = wrapped_xor[14: 114, 14: 114]
                x = cv2.normalize(wrapped_xor, dst=np.zeros((128, 128, 3), dtype=np.uint8), alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX)
                x2 = cv2.normalize(wrapped_xor2, dst=np.zeros((128, 128, 3), dtype=np.uint8), alpha=0, beta=255,
                                   norm_type=cv2.NORM_MINMAX)

                img_pixel.append(x)
                img_pixel.append(x2)
                sub_mask = masks[j]
                # infor = cv2.bitwise_and(comprehensive, comprehensive, mask=sub_mask)
                cv2.imwrite(
                    "gc{}\\xor extracted\\image_{}_cube_{}.jpg".format(a, i, order[j]), x)
                # cv2.imwrite(
                #     "gc{}\\xor extracted\\img_{}_cube_{}.jpg".format(a, i, order[j] + 2), x2)
                img_info.append(img_pixel)
            sub_infor.append(img_info)
        information.append(sub_infor)
    final_infor = []
    print('frame texture extraction finished')
    # prepare for the prediction
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # next step is prediction. But before it we have to classify and filter the image.
    # classify means if this sub video filmed two surfaces of one cube, we have to build two classes and
    # identify one from another, so that we can predict them in classes.
    # filter means
    print('start picking and analysing images')
    for i in range(len(area)):
        num_mask = information[i][0][1]
        # this 'n' is used to judge how many images should we pick from one continuous video
        # from my research, the xor_algorithm has a frequency of 3 Hz, and my video fps is 25,
        # so that I will pick one frame from every 8 frames
        n = int(len(information[i]) / 8)
        img = []
        for j in range(num_mask):
            sub_img128 = []
            sub_img100 = []
            sub_sum = []

            for k in range(len(area[i])):
                if len(information[i][k]) > 2:
                    # print(information[i][k][0])
                    s = 0
                    if num_mask == 1:
                        size128 = information[i][k][2][3]
                        size100 = information[i][k][2][4]
                        size = size128
                        size128_ = cv2.cvtColor(size, cv2.COLOR_BGR2GRAY)
                        sub_img128.append(size128)
                        sub_img100.append(size100)
                        for l in range(len(size128_)):
                            s = s + sum(size128_[l])
                        xcord = information[i][k][2][1]
                        ycord = information[i][k][2][2]

                    if num_mask == 2 and len(information[i][k]) == 4:
                        if information[i][k][2][0] == j:
                            # print('k', k)
                            size128 = information[i][k][2][3]
                            size100 = information[i][k][2][4]
                            size = size128
                            size128_ = list(cv2.cvtColor(size, cv2.COLOR_BGR2GRAY))
                            sub_img128.append(size128)
                            sub_img100.append(size100)
                            for l in range(len(size128_)):
                                s = s + sum(size128_[l])
                            xcord = information[i][k][2][1]
                            ycord = information[i][k][2][2]

                        elif information[i][k][3][0] == j:
                            size128 = information[i][k][3][3]
                            size100 = information[i][k][3][4]
                            size = size128
                            size128_ = list(cv2.cvtColor(size, cv2.COLOR_BGR2GRAY))
                            sub_img128.append(size128)
                            sub_img100.append(size100)
                            for l in range(len(size128_)):
                                s = s + sum(size128_[l])
                            xcord = information[i][k][3][1]
                            ycord = information[i][k][3][2]
                else:
                    pass
                sub_sum.append(s)
            result = np.zeros(7, dtype=np.uint8)
            # only pick the first 'n' images which contain the most strongest signal.
            for k in range(n):
                s1 = sub_sum.index(max(sub_sum))
                sub_sum[s1] = 0
                k = k + 1
                inter = np.array(sub_img128[s1])
                img = inter.astype(np.uint8)
                cv2.imwrite('gc{}\\size_128\\range{}_mask_{}_img{}.jpg'.format(a, i, j, k), img)
                inter2 = np.array(sub_img100[s1])
                img3 = inter2.astype(np.uint8)
                cv2.imwrite('gc{}\\size_100\\100range{}_mask_{}_img{}.jpg'.format(a, i, j, k), img3)
                PIL_image = Image.open('gc{}\\size_100\\100range{}_mask_{}_img{}.jpg'.format(a, i, j, k))

                # neural network prediction
                sub_result = np.array(predict(PIL_image))
                result[sub_result] = result[sub_result] + 1
                # print(result)
            final_result = list(result).index(max(result))
            # show and count the prediction result
            print(i, final_result, result[final_result] / sum(result))
            re = [i, num_mask, xcord, ycord, final_result]

            # class_indict[str(final_result)]
            # draw the predicted label on the image
            for l in range(len(area[i])):
                frame = cv2.imread("gc{}\\orig_{}.jpg".format(a, information[i][l][0]))
                PIL_image = Image.fromarray(frame)
                draw = ImageDraw.Draw(PIL_image)
                # change your font here
                font_ = ImageFont.truetype(font='simhei.ttf', size=50)
                draw.text((xcord, ycord), class_indict[str(final_result)], fill='red', font=font_)
                # PIL_image.save("gc{}\\extracted\\image_{}.jpg".format(a, i))
                PIL_image.save("gc{}\\orig_{}.jpg".format(a, information[i][l][0]))
                # image = np.array(PIL_image)
    # generate prediction video
    for c in range(1, num - 3):
        img_o = cv2.imread("gc{}\\original\\orig_{}.jpg".format(a, c))
        img_o2 = cv2.resize(img_o, (960, 540))
        img_f = cv2.imread('gc{}\\orig_{}.jpg'.format(a, c))
        img_f2 = cv2.resize(img_f, (960, 540))
        img_x = cv2.imread('gc{}\\xor_{}.jpg'.format(a, c))
        img_x2 = cv2.resize(img_x, (960, 540))
        img_show = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        img_show[0:540, 0:960] = img_o2
        img_show[540:1080, 0:960] = img_f2
        img_show[0:540, 960:1920] = img_x2
        outVideo.write(img_show)
print('Video process finished')
