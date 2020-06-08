from django.urls import path

from . import views
from django.urls import path, include
from django.conf.urls import url

urlpatterns = [
    path('register', views.register, name='register'),
    path('login', views.login, name='login'),
    path('signout', views.signout, name='signout'),

]