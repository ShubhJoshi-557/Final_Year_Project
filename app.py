from logging import debug
from flask import Flask, render_template, request
import main
import urllib.request

app = Flask(__name__)

@app.route("/")
def home():
    dirty_url = request.args.get("img_url")
    clean_url = dirty_url.replace("__monkey__","&")
    urllib.request.urlretrieve(clean_url, "static/socialmedia_img.png")
    print(dirty_url)
    return render_template("test_extension_popup.html")

@app.route("/check/")
def calculation():
    main.get_and_save_result("static/socialmedia_img.png","static/final_output.png")
    return "SUCCESS"

@app.route("/fullpage-ss/")
def screenshot_fullpage():
    ss_url = request.args.get("img_url")
    print(ss_url)
    return render_template("show_fullpage_ss.html")

if __name__ == "__main__":
    app.run(debug=True)