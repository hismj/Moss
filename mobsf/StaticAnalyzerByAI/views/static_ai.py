from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.conf import settings
def static_analyzer_ai(request):
    context = {
        'title': 'AI静态分析',
        'version': settings.MOBSF_VER,
        # 添加其他必要的上下文变量
    }
    return render(request, 'static_analysis_by_ai/static_analyzer_by_ai.html', context)
