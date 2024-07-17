from django.http import JsonResponse
from AI_2.pred import predict as apk_predict
import zipfile
from tempfile import TemporaryDirectory
from pathlib import Path


def run_model2(request):
    if request.method == 'POST':
        apk_files = request.FILES.getlist('apk_files')
        if apk_files:
            predictions = []
            for apk_file in apk_files:
                print(f'Uploaded APK file name: {apk_file.name}')

                with TemporaryDirectory() as tmpdirname:
                    tmpdir = Path(tmpdirname)
                    apk_path = tmpdir / apk_file.name
                    with open(apk_path, 'wb') as f:
                        for chunk in apk_file.chunks():
                            f.write(chunk)

                    predicted_label = apk_predict(str(apk_path))

                    predictions.append({
                        'file_name': apk_file.name,
                        'prediction': str(predicted_label),  # 确保 predicted_label 是字符串
                    })

            return JsonResponse({'status': 'success', 'predictions': predictions})
        else:
            return JsonResponse({'status': 'error', 'error': 'No files uploaded.'})
    return JsonResponse({'status': 'error', 'error': 'Invalid request method.'})

