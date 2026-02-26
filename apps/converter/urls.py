from django.urls import path
from . import views

app_name = 'converter'

urlpatterns = [
    path('', views.UploadView.as_view(), name='upload'),
    path('result/<uuid:job_id>/', views.ResultView.as_view(), name='result'),
    path('preview/<uuid:job_id>/', views.PreviewView.as_view(), name='preview'),
    path('debug/<uuid:job_id>/', views.DebugPageView.as_view(), name='debug'),
    path('debug/<uuid:job_id>/overlay/', views.DebugOverlayView.as_view(), name='debug_overlay'),
    path('debug/<uuid:job_id>/data/', views.DebugDataView.as_view(), name='debug_data'),
    path('debug/<uuid:job_id>/compare/', views.DebugCompareDataView.as_view(), name='debug_compare_data'),
    path('api/convert/', views.APIConvertView.as_view(), name='api_convert'),
    path('api/test/', views.APITestFromFolder.as_view(), name='api_test'),
]
