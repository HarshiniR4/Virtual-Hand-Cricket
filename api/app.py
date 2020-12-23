from flask import Flask, render_template, Response, request
from hand_cricket import VideoCamera
app = Flask(__name__, template_folder='D:/HackBattle/templates')

key=''
@app.route('/')
def index():
    # Main page
    return render_template('index.html')

def gen_game(hand_cricket):
    """Video streaming generator function."""
    while True:
        frame= hand_cricket.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        
@app.route('/game')

def game():
    return Response(gen_game(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(debug=False,threaded=False)