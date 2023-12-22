"""
URL configuration for neuronaldeploy project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from network import views
from network import models

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.index, name="index"),
    path("success", views.process_image, name="process_image"),
    path("final/<path:temp_file_path>/", views.final_page, name="final_page")
]
