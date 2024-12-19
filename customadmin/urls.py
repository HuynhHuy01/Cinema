from django.urls import path
from . import views


urlpatterns = [
    path('edit/',views.edit_cinema,name='edit-cinema'),
    path('index/',views.index,name='index'),
    path('singup/',views.signup,name='admin-signup'),
    path('login/',views.login,name='admin-login'),
    path('menu/',views.menu,name='menu'),
    path('profile/',views.edit_profile,name='profile'),
    path('result/',views.edit_result,name='result'),
    path('seat/',views.edit_seat,name='seat'),
    path('movie/',views.edit_movies,name='movie'),
    path('reset/',views.reset,name='reset'),
    path('adduser/',views.add_user,name='add-user'),
    path('edituser/',views.edit_user,name='edit-user'),
    path('indexuser/',views.user_index,name='user-index'),
    path('addmovie/',views.add_movie,name='add-cinema'),
    path('editmovie/',views.edit_movie,name='edit-cinema'),
    path('indexmovie/',views.movie_index,name='user-cinema'),
    
]