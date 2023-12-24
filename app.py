from flask import Flask, request, render_template, Response, g
from flask_cors import CORS, cross_origin
from test import check_if_performed as check_if_performed_test, cv2

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


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
#     g.video_camera = VideoCamera()


def main():
    # app.run()
    app.run(debug=True)


@app.route("/check_if_performed", methods=["GET"])
@cross_origin()
def check_if_performed():
    print(request.url)
    print("------------------ requested ------------------")
    print("------------------ requested ------------------")
    print("------------------ requested ------------------")

    action = request.args.get("action")
    if action == "" or action == None:
        print("Action not provided")
        return "Please provide an action", 400

    try:
        return check_if_performed_test(action)
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
