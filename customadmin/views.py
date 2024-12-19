from django.shortcuts import render

# Create your views here.
def edit_cinema(request):
    return render(request,"edit.html")

def index(request):
    return render(request,"admincustom.html")

def login(request):
    return render(request,"login.html")

def signup(request):
    return render(request,"Signup.html")

def menu(request):
    return render(request,"menu.html")

def edit_movies(request):
    return render(request,"movies.html")

def edit_profile(request):
    return render(request,"profile.html")

def edit_result(request):
    return render(request,"results.html")

def edit_seat(request):
    return render(request,"seat_booking.html")

def reset(request):
    return render(request,"resetpassword.html")


def add_user(request):
    return render(request,"user/add.html")

def edit_user(request):
    return render(request,"user/edit.html")

def user_index(request):
    return render(request,"user/index.html")

def add_movie(request):
    return render(request,"cinema/add.html")

def edit_movie(request):
    return render(request,"cinema/edit.html")

def movie_index(request):
    return render(request,"cinema/index.html")