from flask import Flask, render_template, Response, request ,jsonify
#Import OpenCv package
import cv2
from ultralytics import YOLO
# import numpy as np
import numpy as np
import os
import math


app = Flask(__name__)

camera = None
is_running = False
detected_sign = "None"
confidence = 0

def generate1():
    model=YOLO('best_american.pt')
    classNames=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    global camera, is_running, detected_sign ,confidence
    while is_running and camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        else:
            results=model(frame,stream=True)

        for r in results:
            boxes=r.boxes

            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                confidence=math.ceil((box.conf[0]*100))/100
                print("confidence-->",confidence)
                cls=int(box.cls[0])
                print("class name-->",classNames[cls])
                detected_sign = classNames[cls]

                org=[x1,y1]
                font=cv2.FONT_HERSHEY_SIMPLEX
                fontscale=1
                color=(0,255,0)
                thickness=2

                cv2.putText(frame,classNames[cls],org,font,fontscale,color,thickness)
                        
                        
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            frame = encodedImage.tobytes()
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    
            
def generate2():
    model=YOLO('best_indian.pt')
    classNames=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    global camera, is_running, detected_sign, confidence
    while is_running and camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        else:
            results=model(frame,stream=True)

        for r in results:
            boxes=r.boxes

            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                confidence=math.ceil((box.conf[0]*100))/100
                print("confidence-->",confidence)
                cls=int(box.cls[0])
                print("class name-->",classNames[cls])
                detected_sign = classNames[cls]
                org=[x1,y1]
                font=cv2.FONT_HERSHEY_SIMPLEX
                fontscale=1
                color=(0,255,0)
                thickness=2

                cv2.putText(frame,classNames[cls],org,font,fontscale,color,thickness)
                        
                        
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            frame = encodedImage.tobytes()
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    
            

@app.route("/home")
def home():
    return render_template('logo.html')

@app.route('/start', methods=['POST'])
def start():
    global camera, is_running
    if not is_running:
        camera = cv2.VideoCapture(0)  # open webcam
        is_running = True
    return "Started"

@app.route('/stop', methods=['POST'])
def stop():
    global camera, is_running
    if is_running:
        is_running = False
        if camera:
            camera.release()
    return "Stopped"

@app.route("/video_feed1")
def video_feed1():
    return Response(generate1(),mimetype='multipart/x-mixed-replace; boundary=frame')  

@app.route("/asl")
def asl():
    return render_template('asl.html')
    
@app.route("/video_feed2")
def video_feed2():
    return Response(generate2(),mimetype='multipart/x-mixed-replace; boundary=frame')      
    
@app.route("/isl")
def isl():
    return render_template('isl.html')    
            
@app.route("/details")
def details():
    return render_template('index.html') 
    
@app.route("/help")
def help():
    return render_template('hel.html') 

@app.route('/get_output')
def get_output():
    global detected_sign, confidence
    return jsonify({"sign": detected_sign, "confidence": confidence})
    
if __name__ == "__main__":
    app.run(debug=True)

