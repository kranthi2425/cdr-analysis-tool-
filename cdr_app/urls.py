from django.urls import path
from . import views

urlpatterns = [
    path('', views.cdr_list, name='cdr_list'),
]