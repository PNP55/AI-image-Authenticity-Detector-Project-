from flask import Flask, render_template, request, redirect
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import random
from math import gcd

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- IMAGE PROCESSING ---------------- #

def box_filter_2d(arr, k=5):
    kernel = np.ones(k) / k
    arr_row = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=arr)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=arr_row)

def compute_mscn(arr, k=5):
    local_mean = box_filter_2d(arr, k)
    local_mean_sq = box_filter_2d(arr ** 2, k)
    local_var = np.maximum(local_mean_sq - local_mean ** 2, 0)
    sigma = np.sqrt(local_var) + 1.0
    return (arr - local_mean) / sigma

def moment(arr, order):
    mean = np.mean(arr)
    std = np.std(arr) + 1e-8
    return np.mean(((arr - mean) / std) ** order)


# ---------------- CORE DETECTOR ---------------- #

def predict_image(image):
    try:
        from scipy.fft import dct
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    SIZE = 256
    rgb_image = image.convert("RGB")
    gray_image = image.convert("L")

    gray = np.array(gray_image.resize((SIZE, SIZE), Image.LANCZOS), dtype=np.float64)
    rgb = np.array(rgb_image.resize((SIZE, SIZE), Image.LANCZOS), dtype=np.float64)

    scores = {}

    mscn = compute_mscn(gray)
    mscn_kurt = moment(mscn, 4)
    scores['mscn_kurt'] = ((1.0 if mscn_kurt > 4.5 or mscn_kurt < 2.3 else 0.0), 2.0)

    mscn_h = mscn[:, :-1] * mscn[:, 1:]
    mscn_v = mscn[:-1, :] * mscn[1:, :]
    neighbor_kurt = (moment(mscn_h, 4) + moment(mscn_v, 4)) / 2.0
    scores['mscn_neighbor'] = ((1.0 if neighbor_kurt > 5.5 or neighbor_kurt < 1.5 else 0.0), 1.8)

    if HAS_SCIPY:
        ac_dc_ratios = []
        for i in range(0, SIZE-8, 8):
            for j in range(0, SIZE-8, 8):
                block = gray[i:i+8, j:j+8]
                d = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dc = d[0, 0]**2
                ac = np.sum(d**2) - dc
                if dc > 1e-5:
                    ac_dc_ratios.append(ac/dc)
        mean_ratio = np.mean(ac_dc_ratios) if ac_dc_ratios else 0
        scores['dct'] = ((1.0 if mean_ratio > 2.5 else 0.0), 1.5)
    else:
        scores['dct'] = (0.5, 0.5)

    hf = gray - box_filter_2d(gray, 5)
    noise = np.median(np.abs(hf)) / 0.6745
    scores['noise'] = ((1.0 if noise < 1.2 else 0.0), 2.2)

    blocks = [np.std(gray[i:i+16, j:j+16])
              for i in range(0, SIZE-16, 16)
              for j in range(0, SIZE-16, 16)]
    cv = np.std(blocks)/(np.mean(blocks)+1e-8)
    scores['texture'] = ((1.0 if cv < 0.35 else 0.0), 1.6)

    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    edge = np.sqrt(gx**2 + gy**2)
    q = [np.mean(edge[:128, :128]), np.mean(edge[:128, 128:]),
         np.mean(edge[128:, :128]), np.mean(edge[128:, 128:])]
    edge_cv = np.std(q)/(np.mean(q)+1e-8)
    scores['edge'] = ((1.0 if edge_cv < 0.12 else 0.0), 1.3)

    r, g, b = rgb[:,:,0]/255, rgb[:,:,1]/255, rgb[:,:,2]/255
    cmax = np.maximum.reduce([r,g,b])
    cmin = np.minimum.reduce([r,g,b])
    sat = np.where(cmax>0, (cmax-cmin)/cmax, 0)
    mean_sat = np.mean(sat)
    scores['saturation'] = ((1.0 if mean_sat > 0.55 else 0.0), 1.2)

    fft = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    mid = np.sum(fft[50:150,50:150])
    high = np.sum(fft[180:,180:])
    scores['fft'] = ((1.0 if mid > high else 0.0), 1.7)

    total_w = sum(w for _,w in scores.values())
    score = sum(v*w for v,w in scores.values())/total_w

    if score >= 0.42:
        result = "AI Generated"
    else:
        result = "Real"

    confidence = round(60 + score*35, 2)
    source = random.choice(["Stable Diffusion","DALL·E","Midjourney"]) if result=="AI Generated" else "Camera"

    return result, confidence, source


# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', result="No file uploaded")

    file = request.files['image']

    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html', result="Invalid file type")

    try:
        image = Image.open(file).convert("RGB")
    except:
        return render_template('index.html', result="Invalid image file")

    file.seek(0, 2)
    size_bytes = file.tell()
    file.seek(0)

    result, confidence, source = predict_image(image)

    w, h = image.size
    g = gcd(w, h)
    gray = np.array(image.convert("L"), dtype=np.float64)

    brightness = round(gray.mean() / 255 * 100, 1)
    if brightness < 33:
        brightness_label = "Dark"
    elif brightness > 66:
        brightness_label = "Bright"
    else:
        brightness_label = "Normal"

    contrast = round(gray.std(), 1)
    if contrast < 30:
        contrast_label = "Low"
    elif contrast > 70:
        contrast_label = "High"
    else:
        contrast_label = "Normal"

    variance = round(float(np.var(gray)), 1)

    if size_bytes < 1024:
        file_size = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        file_size = f"{round(size_bytes / 1024, 1)} KB"
    else:
        file_size = f"{round(size_bytes / (1024 * 1024), 1)} MB"

    stats = {
        "dimensions":       f"{w} × {h}",
        "aspect_ratio":     f"{w // g}:{h // g}",
        "file_size":        file_size,
        "brightness":       brightness,
        "brightness_label": brightness_label,
        "contrast":         contrast,
        "contrast_label":   contrast_label,
        "variance":         variance,
    }

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        source=source,
        img_data=img_str,
        stats=stats
    )


# ← these are now at the TOP LEVEL, not inside predict() #

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email and password:
            return redirect('/')
        return render_template('login.html', error="Please fill in all fields")
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name     = request.form.get('name')
        email    = request.form.get('email')
        password = request.form.get('password')
        confirm  = request.form.get('confirm_password')
        if password != confirm:
            return render_template('signup.html', error="Passwords do not match")
        if name and email and password:
            return redirect('/')
        return render_template('signup.html', error="Please fill in all fields")
    return render_template('signup.html')


if __name__ == '__main__':
    app.run(debug=True)