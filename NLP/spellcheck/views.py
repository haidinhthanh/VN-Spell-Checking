from django.shortcuts import render
from django.template import loader
import spellcheck.vnspellcheck
import json
from django.http import HttpResponse,JsonResponse 
# Create your views here.
def index(request):
		return render(request,'./index.html', {})
def spellcheck_predic(request):
	uptxt= request.POST['uptxt']
	txt_predict= spellcheck.vnspellcheck.predict(uptxt)
	return HttpResponse(txt_predict)