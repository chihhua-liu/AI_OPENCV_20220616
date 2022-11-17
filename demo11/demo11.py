#
# from flask import Flask, render_template, Response
# import cv2
#
# app = Flask(__name__)
#
#
# @app.route("/")
# def index():
#     return render_template("index.html")
#
#
# def generate():
#     global outputFrame, lock
#     while True:
#         with lock:
#             if outputFrame is None:
#                 continue
#             flag, encodedImage = cv2.imencode("*.jpg", outputFrame)
#             if not flag:
#                 continue
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage)
#                + b'\r\n')
#
#
# @app.route("/video_feed")
# def video_feed():
#     return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

#----------------------------------------------------------------------
from flask import Flask, render_template, Response
import cv2
import threading
from imutils.video import VideoStream
import imutils
import datetime

app = Flask(__name__)

outputFrame = None
lock = threading.Lock()
# default == 0
# instructor ==1
vs = VideoStream(src=0).start()


def getScreen(frameCount):
    global vs, outputFrame, lock
    total = 0
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp = datetime.datetime.now()
        cv2.putText(gray, timestamp.strftime("%A %d %B %Y %I:%M:%S:%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 0, 255), 1)
        total += 1
        with lock:
            outputFrame = gray.copy()


@app.route("/")
def index():
    return render_template("index.html")


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            flag, encodedImage = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage)
               + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    t = threading.Thread(target=getScreen, args=(32,))
    t.daemon = True
    t.start()
    app.run(debug=True, threaded=True, use_reloader=False)
vs.stop()