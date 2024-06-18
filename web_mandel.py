import cuda_mandel
from flask import Flask, request, render_template
from flask.wrappers import Response
from time import sleep
from timeit import default_timer as timer
import cv2
import threading


port = 12345
app = Flask(__name__)


def generate():
    while True:
        start_time = timer()

        success, buf = cv2.imencode(".jpg", cuda_mandel.color_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            continue

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )

        # Limit FPS to prevent the browser from choking
        time_passed = timer() - start_time
        time_remains = (1 / cuda_mandel.max_FPS) - time_passed
        if time_remains > 0:        
            sleep(time_remains)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/keypress", methods=["POST"])
def keypress():
    data = request.get_json()
    key = data["key"]

    s = cuda_mandel.state
    m = cuda_mandel.Move
    z = cuda_mandel.Zoom
    if key == "j":
        s.zoom = z.NONE if s.zoom == z.IN else z.IN
    elif key == "k":
        s.zoom = z.NONE if s.zoom == z.OUT else z.OUT
    elif key == "a":
        s.move = m.NONE if s.move == m.LEFT else m.LEFT
    elif key == "d":
        s.move = m.NONE if s.move == m.RIGHT else m.RIGHT
    elif key == "w":
        s.move = m.NONE if s.move == m.UP else m.UP
    elif key == "s":
        s.move = m.NONE if s.move == m.DOWN else m.DOWN
    elif key == "l":
        s.zoom = z.NONE
        s.move = m.NONE

    print(f"Received keypress: {key}")
    return "OK", 200


def start_flask_app():
    app.run(port=port, debug=True)


if __name__ == "__main__":
    thread = threading.Thread(target=cuda_mandel.run).start()
    start_flask_app()
