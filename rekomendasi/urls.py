
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('fuzzy/', views.fuzzy_test, name='fuzzy'),
    path('machine-learning/', views.machine_learning, name='machine_learning'),
    path('ai-assistant/', views.ai_assistant, name='ai_assistant'),
    path('visualisasi/', views.visualisasi, name='visualisasi'),
]

def visualisasi(request):
    return render(request, 'rekomendasi/visualisasi.html')