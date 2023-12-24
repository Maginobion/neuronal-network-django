from django import forms
import network.models as models

class ImageForm(forms.Form):
    image = forms.FileField() 
    class Meta:        
        fields = ['image']