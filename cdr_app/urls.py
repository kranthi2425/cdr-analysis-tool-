from django.urls import path
from . import views
# from .views import visualization

urlpatterns = [
    path('', views.cdr_list, name='cdr_list'),
    path('visualization/', views.cdr_visualization, name='cdr_visualization'),
]