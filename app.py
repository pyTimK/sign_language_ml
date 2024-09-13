from flask import Flask, request, render_template, Response, g
from flask_cors import CORS, cross_origin
from test import check_if_performed as check_if_performed_test, cv2
import json
from flask_caching import Cache
from flask_socketio import SocketIO, emit

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
cache = Cache(app, config={"CACHE_TYPE": "simple"})
socketio = SocketIO(app, debug=True, cors_allowed_origins="*")
socketio.init_app(app)


@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


@socketio.on("chat")
def handle_chat(data):
    # print(data)
    image_base64: str = data["img"]
    action: str = data["action"]

    if action == "" or action == None:
        print("Action not provided")
        return "Please provide an action", 400

    try:
        # print(image_base64)
        print(action)
        data, status = check_if_performed_test(action, image_base64, cache)
        emit("chat", data, broadcast=True)
        # check_if_performed_test(g.video_camera.video, action)
        # if result == True:
        #     return {"success": True, "error": ""}, 200
        # else:
        #     return {"success": False, "error": ""}, 200

    except Exception as e:
        print(e)
        emit("chat", {"error": "Not Found", "success": False}, broadcast=True)

    # emit to chat


# with app.app_context():
#     g.sequence = []


# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#     def __del__(self):
#         self.video.release()

#     def get_frame(self):
#         ret, frame = self.video.read()
#         ret, jpeg = cv2.imencode(".jpg", frame)
#         return jpeg.tobytes()


# @app.before_request
# def before_request():
#     print("------------------ before_request ------------------")
#     print(hasattr(g, "sequence"))
#     print("------------------ before_reques2t ------------------")
#     g.sequence = []
#     print(hasattr(g, "sequence"))
#     print("------------------ before_reques2t ------------------")
#     print(hasattr(g, "sequence"))
#     if g.sequence is None:
#         g.sequence = []


def main():
    # app.run()
    # app.run(debug=True)
    socketio.run(app, debug=True)


@app.route("/check_if_performed", methods=["POST"])
@cross_origin()
def check_if_performed():
    print(request.url)
    print("------------------ requested ------------------")
    print("------------------ requested ------------------")
    print("------------------ requested ------------------")

    # Access the raw request body
    raw_data = request.data
    # Convert the raw data to a string (assuming it's in utf-8 encoding)
    data_as_string = raw_data.decode("utf-8")
    image_base64: str = json.loads(data_as_string)["img"]

    action = request.args.get("action")
    if action == "" or action == None:
        print("Action not provided")
        return "Please provide an action", 400

    try:
        # print(image_base64)
        print(action)
        return check_if_performed_test(action, image_base64, cache)
        # check_if_performed_test(g.video_camera.video, action)
        # if result == True:
        #     return {"success": True, "error": ""}, 200
        # else:
        #     return {"success": False, "error": ""}, 200

    except Exception as e:
        print(e)
        return {"error": "Not Found", "success": False}, 404


# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


# @app.route("/video_feed")
# def video_feed(camera):
#     return Response(gen(camera), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    main()
