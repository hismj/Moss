from django.http import JsonResponse, HttpResponse
import requests
import os

def download_apk(request):
    if request.method == 'POST':
        url = request.POST.get('apk_url')
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            file_name = os.path.basename(url)
            response_content = response.content
            response = HttpResponse(response_content, content_type='application/vnd.android.package-archive')
            response['Content-Disposition'] = f'attachment; filename={file_name}'
            return response
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': '请求方法不正确'})

