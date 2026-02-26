from django import forms


class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label='Upload Screenshot',
        help_text='Upload a screenshot of a dashboard or web page'
    )
    languages = forms.MultipleChoiceField(
        choices=[
            ('en', 'English'),
            ('fa', 'Persian (Farsi)'),
        ],
        initial=['fa', 'en'],
        widget=forms.CheckboxSelectMultiple,
        required=False,
        label='Text Languages',
    )
