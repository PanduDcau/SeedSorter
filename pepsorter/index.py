import os
from flask import render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from app import app
from main import process_uploaded_image


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


def handle_file_upload(request):
    if 'file' not in request.files:
        return None, {'error': 'No file part'}
    file = request.files['file']
    if file.filename == '':
        return None, {'error': 'No file selected for uploading'}
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return filename, None
    return None, {'error': 'Invalid request method'}


@app.route('/processImage', methods=['POST'])
def process_image_route():
    filename, error = handle_file_upload(request)
    if error:
        return jsonify(error)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result_path, purity, pepper_seed_count, total_count = process_uploaded_image(file_path)

    if result_path:
        # Generate the final image URL from the output folder
        result_url = f"{request.host_url}output/{os.path.basename(result_path)}"
    else:
        result_url = None

    return jsonify({
        'result': result_url,
        'purity': purity,
        'pepper_seed_count': pepper_seed_count,
        'total_count': total_count
    })


@app.route('/output/<filename>', methods=['GET'])
def result_output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True,host='192.168.1.10', port=4000)
