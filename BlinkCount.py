import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

videoCapture = cv2.VideoCapture('Video1.mp4')
faceDetector = FaceMeshDetector(maxFaces=1)
livePlotGraph = LivePlot(640, 360, [20, 50], invert=True)

keypoints = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
blinkRatios = []
blinkCount = 0
frameCount = 0
indicatorColor = (255, 0, 255)

while True:
    if videoCapture.get(cv2.CAP_PROP_POS_FRAMES) == videoCapture.get(cv2.CAP_PROP_FRAME_COUNT):
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = videoCapture.read()
    frame, detectedFaces = faceDetector.findFaceMesh(frame, draw=False)

    if detectedFaces:
        primaryFace = detectedFaces[0]
        for kp in keypoints:
            cv2.circle(frame, primaryFace[kp], 5, indicatorColor, cv2.FILLED)

        topLeft = primaryFace[159]
        bottomLeft = primaryFace[23]
        farLeft = primaryFace[130]
        farRight = primaryFace[243]
        verticalDist, _ = faceDetector.findDistance(topLeft, bottomLeft)
        horizontalDist, _ = faceDetector.findDistance(farLeft, farRight)

        cv2.line(frame, topLeft, bottomLeft, (0, 200, 0), 3)
        cv2.line(frame, farLeft, farRight, (0, 200, 0), 3)

        blinkRatio = int((verticalDist / horizontalDist) * 100)
        blinkRatios.append(blinkRatio)
        if len(blinkRatios) > 3:
            blinkRatios.pop(0)
        averageRatio = sum(blinkRatios) / len(blinkRatios)

        if averageRatio < 35 and frameCount == 0:
            blinkCount += 1
            indicatorColor = (0, 200, 0)
            frameCount = 1
        if frameCount != 0:
            frameCount += 1
            if frameCount > 10:
                frameCount = 0
                indicatorColor = (255, 0, 255)

        cvzone.putTextRect(frame, f'Blink Count: {blinkCount}', (50, 100), colorR=indicatorColor)

        plotImage = livePlotGraph.update(averageRatio, indicatorColor)
        frame = cv2.resize(frame, (640, 360))
        stackedImage = cvzone.stackImages([frame, plotImage], 2, 1)
    else:
        frame = cv2.resize(frame, (640, 360))
        stackedImage = cvzone.stackImages([frame, frame], 2, 1)

    cv2.imshow("Image", stackedImage)
    cv2.waitKey(25)
