import cv2
import numpy as np
# import picamera
import os
import io
import time
import multiprocessing

low_threshold = 0
high_threshold = 150
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 200  # minimum number of votes (intersections in Hough grid cell)
max_line_gap = 20  # maximum gap in pixels between connectable line segments

def get_interest(img) : # 라이터 위치를 찾기 위한 이미지의 절반을 흑백 처리 함수
    img[0:206, :] = 0
    return img

def checkRawRatio(candidate) :  # 라이터 고정대를 찾기 위한 좌표 추정 함수
    return int(candidate * (170/268))

def checkHeadRatio(raw, stick) :    # 라이터 헤드를 찾기 위한 좌표 추정 함수
    return int((stick-raw) * (5/98))

def findRaw(img) :  # 라이터 고정대 좌표를 찾기 위한 함수
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = get_interest(gray)
    kernel_size = 5

    for i in range(5) :
        gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    min_line_length = int(img.shape[0]*0.4)  # minimum number of pixels making up a line
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    candidate = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if y1 > img.shape[1]*0.55 :
                candidate.append([y1, y2])

    if candidate :
        candidate.sort(reverse=True, key = lambda x : x[0])
        return checkRawRatio(candidate[0][0]), candidate[0][0]
    else :
        return -1, -1

def getCapture(cap) :   # 반복적으로 화면 캡쳐를 얻는 함수
    # 로컬에 화면 캡쳐 이미지를 저장함
    with picamera.PiCamera() as camera :
        camera.resolution = (416, 416)
        while True :
            camera.capture("images/"+str(cap)+".jpg")
            cap += 1

def yolo(cap) :     # 로컬에 저장된 화면 캡쳐를 불러와 라이터의 스티커 불량 여부를 확인하는 함수
    # 인식이 완료된 화면 캡쳐는 삭제 됨
    raw = 0
    
    net = cv2.dnn.readNet("yolov3-tiny_4000.weights", "yolov3-tiny.cfg")    # 학습 모델을 불러옴
    classes = ["Head", "Body"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    stickers = []
    for i in range(10) :    # 스티커 불량 여부를 판정하기 위한(템플릿 매칭에 사용될) 기준 스티커를 불러옴
        if os.path.isfile('num'+str(i)+'.jpg') :
            stickers.append(cv2.imread('num'+str(i)+'.jpg'))

    prev = time.time()
    
    while True :
        if os.path.isfile("images/"+str(cap)+".png") :      # 로컬에 저장된 화면 캡쳐를 불러옴
            img = cv2.imread("images/"+str(cap)+".png")
            try :
                temp, stick = findRaw(img)      # 라이터 위치를 특정하기 위한 받침대 위치 확인
                if temp > 0 :
                    if 0.9*raw < temp < 1.1*raw : print("카메라가 위치를 벗어남")   # 이전 프레임과 비교하여 받침대 위치가 벗어나면 카메라가 움직인 것
                    raw = temp

                #-----라이터 헤드를 찾고 헤드를 기준으로 바디를 추정-----#

                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                confidences = []
                boxes = []
                
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.3:
                            # Object detected
                            center_x = int(detection[0] * 416)
                            center_y = int(detection[1] * 416)
                            w = int(detection[2] * 416)
                            h = int(detection[3] * 416)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            # 바디를 학습시키지 않는다는 가정 하에
                            if y+h < raw * 1.05 :
                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

                # 인식된 라이터가 다섯개 미만이면 화면 캡쳐가 흔들린 것 혹은 라이터가 아래로 내려간 상태인 것으로 간주
                if len(boxes) < 5 :
                    continue

                boxes.sort()
                
                #-----라이터 헤드가 가려진 경우 라이터 헤드의 위치와 그에 따른 바디 위치를 임의로 추정-----#

                if len(boxes) < 10 :
                    first = boxes[0]
                    last = boxes[-1]
                    between = checkHeadRatio(raw, stick)
                    i = 0
                    while i < len(boxes)-1 :
                        if boxes[i+1][0] - boxes[i][0] + boxes[i][2] > between :
                            numOfTempLighter = (boxes[i+1][0] - (boxes[i][0] + boxes[i][2] + between)) // (between + boxes[i][2])
                            if numOfTempLighter > 0 :
                                for k in range(numOfTempLighter) :
                                    boxes.append([(boxes[i][0] + boxes[i][2])+(k+1)*between+k*boxes[i][2], boxes[i][1], boxes[i][2], boxes[i][3]])
                                i += numOfTempLighter + 1
                                continue
                        i += 1
                    if len(boxes) < 10 :
                        num = 10-len(boxes)
                        for k in range(num) :
                            if first[0] - (k+1)*(between + first[2]) < 0 : break
                            if (last[0] + last[2])+(k+1)*between+k*last[2] > 416 : break
                            boxes.append([first[0] - (k+1)*(between + first[2]), first[1], first[2], first[3]])
                            boxes.append([(last[0] + last[2])+(k+1)*between+k*last[2], last[1], last[2], last[3]])

                #-----스티커의 불량 여부 확인-----#

                # for index in boxes :
                #     cv2.rectangle(img, (index[0], index[1]), (index[0]+index[2], index[1]+index[3]), (255, 0, 0), 1, cv2.LINE_8)
                results = []
                for index in boxes :
                    cut_img = img[index[1]+index[3]:stick, index[0]:index[0]+index[2]]
                    resul = []
                    for sticker in stickers :
                        sticker = cv2.resize(sticker, dsize=(0, 0), fx=(cut_img.shape[1]/sticker.shape[1]), fy=(cut_img.shape[1]/sticker.shape[1]), interpolation=cv2.INTER_LINEAR)
                        result = cv2.matchTemplate(cut_img, sticker, cv2.TM_SQDIFF_NORMED)

                        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
                        x, y = minLoc
                        h, w, c = sticker.shape
                        # cv2.rectangle(cut_img, (x, x+10), (y, y+10), (255, 255, 0), 2, cv2.LINE_8)
                        # cv2.imshow("화면2", sticker)
                        # cv2.imshow("화면", cut_img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        resul.append([index[0]+x, index[1]+index[3]+y, w, h, minVal])
                    resul.sort(key = lambda x : x[4])
                    if resul[0][4] < 0.5 : results.append(resul[0])

                #-----불량이 있을 경우 불량임을 알린다-----#

                if len(results) < 10 : print("불량 있음")

                #-----확인한 불량 여부 가능성과 스티커 위치에 박스 표시-----#

                # for i, index in enumerate(results) :
                #     cv2.putText(img, "%.2f" % index[4], (index[0], index[1]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                #     cv2.rectangle(img, (index[0], index[1]), (index[0]+index[2], index[1]+index[3]), (255, 0, 0), 1, cv2.LINE_8)
                # cv2.imshow("화면", img)

                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # 처리가 끝난 이미지는 무조건 삭제
                os.remove(str(cap)+".jpg")
                cap += 1
                prev = time.time()

            except Exception as e :
                print(str(e))
            
        else :      # 10초 이상 화면 캡쳐가 추가되지 않으면 종료
            if time.time() - prev > 10 :
                return
            else :
                pass
        
if __name__ == '__main__' :
    cap = 0
    proc1 = multiprocessing.Process(target=getCapture, args=(cap,))
    proc1.start()
    proc2 = multiprocessing.Process(target=yolo, args=(cap,))
    proc2.start()
    
    proc1.join()
    proc2.join()