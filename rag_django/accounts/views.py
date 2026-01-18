from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serialization import SignupSerializer
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect


class SignupView(APIView):
    def post(self, request):
        serializer = SignupSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response({"message": "User created"}, status=201)
    
def home(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    return redirect('login')

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST["password"]

        user = authenticate(username=username, password=password)
        if user:
            login(request, user)
            return redirect("dashboard")

    return render(request, "accounts/login.html")

def signup_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        if User.objects.filter(username = username).exists():
            return render(request, 'accounts/signup.html', 
                          {'error': 'Username already exists'})
        
        User.objects.create_user(username= username,
                                 password= password)
        
        return redirect("login")
    
    return render(request, "accounts/signup.html")

def logout_view(request):
    logout(request)
    return redirect("login")
