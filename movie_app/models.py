from django.db import models
from django.db.models import Model, QuerySet
from django.urls import reverse
from django.utils.text import slugify

from django import forms

from user_app.models import User
from datetime import timedelta, datetime
import random
import string

# TODO add rating
class Country(Model):
    country = models.CharField(max_length=75)

    def __str__(self):
        return self.country


class Language(Model):
    lang = models.CharField(max_length=75)

    def __str__(self):
        return self.lang


class Date(Model):
    date = models.DateField()
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.date.year}"

    def get_absolute_url(self):
        return reverse('year-filter-page', args=[self.date.year])


class Genre(Model):
    genre = models.CharField(max_length=50)
    is_active = models.BooleanField(default=True)

    def get_absolute_url(self):
        return reverse('genre-filter-page', args=[self.genre])

    def __str__(self):
        return self.genre


class Quality(Model):
    quality = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.quality


class Serie(Model):
    name = models.CharField(max_length=300)
    imdb = models.FloatField(null=True, blank=True)
    director = models.CharField(max_length=50)
    banner = models.FileField(upload_to='uploads/banners')
    release_date = models.ForeignKey(to=Date, on_delete=models.CASCADE, related_name='release')
    end_date = models.ForeignKey(to=Date, on_delete=models.CASCADE, related_name='end')
    genre = models.ManyToManyField(to=Genre, related_name='serie_genre')
    cast = models.CharField(max_length=200, null=True, blank=True)
    background = models.ImageField(upload_to='uploads/backgrounds', blank=True, null=True)
    is_active = models.BooleanField(default=True)
    description = models.TextField(null=True)
    country = models.ForeignKey(to=Country, null=True, blank=True, on_delete=models.CASCADE)
    lang = models.ForeignKey(to=Language, null=True, blank=True, on_delete=models.CASCADE)
    trailer = models.FileField(upload_to='uploads/trailers', null=True, blank=True)
    slug = models.SlugField(db_index=True, blank=True)

    def get_absolute_url(self):
        return reverse('serie-page', args=[self.slug])

    def __str__(self):
        return f'{self.name}-{self.release_date}/{self.end_date}'

    def save(self, *args, **kwargs):
        self.slug = slugify(self.name)
        super(Serie, self).save(*args, **kwargs)


class Season(Model):
    serie = models.ForeignKey(to=Serie, on_delete=models.CASCADE)
    parts_count = models.IntegerField()
    part_avg_length = models.IntegerField()
    number = models.IntegerField()
    quality = models.ForeignKey(to=Quality, on_delete=models.CASCADE)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.serie} - {self.quality} - S{self.number} "


class Part(Model):
    file = models.FileField(upload_to='uploads/series')
    season = models.ForeignKey(to=Season, on_delete=models.CASCADE)
    number = models.IntegerField()
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.season} E{self.number}"


class Film(Model):
    name = models.CharField(max_length=300)
    length = models.IntegerField()
    imdb = models.FloatField(null=True, blank=True)
    director = models.CharField(max_length=50)
    banner = models.ImageField(upload_to='uploads/banners')
    release_date = models.ForeignKey(to=Date, on_delete=models.CASCADE)
    genre = models.ManyToManyField(to=Genre, related_name='film_genre')
    background = models.ImageField(upload_to='uploads/backgrounds', blank=True, null=True)
    cast = models.CharField(max_length=200, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    description = models.TextField(null=True)
    country = models.ForeignKey(to=Country, null=True, blank=True, on_delete=models.CASCADE)
    lang = models.ForeignKey(to=Language, null=True, blank=True, on_delete=models.CASCADE)
    trailer = models.FileField(upload_to='uploads/trailers', blank=True, null=True)
    slug = models.SlugField(db_index=True, blank=True)

    def get_absolute_url(self):
        return reverse('film-page', args=[self.slug])

    def __str__(self):
        return f'{self.name} - {self.release_date}'

    def save(self, *args, **kwargs):
        self.slug = slugify(self.name)
        super(Film, self).save(*args, **kwargs)


class FilmByQuality(Model):
    film = models.ForeignKey(to=Film, on_delete=models.CASCADE)
    file = models.FileField(upload_to="uploads/films")
    quality = models.ForeignKey(to=Quality, on_delete=models.CASCADE)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.film} - {self.quality}"


class SerieVisit(Model):
    serie = models.ForeignKey(to=Serie, on_delete=models.CASCADE)
    ip = models.CharField(max_length=30)
    user = models.ForeignKey(to='user_app.User', null=True, blank=True, on_delete=models.CASCADE)

    def __str__(self):
        if self.user is not None:
            name = f'{self.serie} - {self.ip} - {self.user.username}'
        else:
            name = f'{self.serie} - {self.ip}'

        return name


class FilmVisit(Model):
    film = models.ForeignKey(to=Film, on_delete=models.CASCADE)
    ip = models.CharField(max_length=30)
    user = models.ForeignKey(to='user_app.User', null=True, blank=True, on_delete=models.CASCADE)

    def __str__(self):
        name = ''
        if self.user is not None:
            name = f'{self.film} - {self.ip} - {self.user.username}'
        elif self.user is None:
            name = f'{self.film} - {self.ip}'

        return name
    
from decimal import Decimal



from django.core.exceptions import ValidationError

class Shows(models.Model):
    shows=models.AutoField(primary_key=True)
    film = models.ForeignKey(to=Film, on_delete=models.CASCADE,related_name='movie_show',null=True,blank=True)
    serie = models.ForeignKey(to=Serie, on_delete=models.CASCADE,related_name='serie_show',null=True,blank=True)
    screen = models.CharField(max_length=300,default="Screen 1")
    start_time= models.TimeField(null=True, blank=True)
    end_time = models.TimeField(null=True, blank=True)
    date =models.CharField(max_length=15, default="")
    price = models.DecimalField(max_digits=10, decimal_places=3, default=Decimal('0.000'))
   
    def clean(self):
        # Kiểm tra nếu cả 'film' và 'serie' đều có giá trị
        if self.film and self.serie:
            raise ValidationError("You can only select either a film or a serie, not both.")
        if not self.film and not self.serie:
            raise ValidationError("You must select either a film or a serie.")

    def save(self, *args, **kwargs):
        # Chia giá trị cho 1000 trước khi lưu để giá trị lưu là 90.000 thay vì 90000.000
        if self.price >= 1000:
            self.price = self.price / Decimal('1000')


        self.clean()    
        super().save(*args, **kwargs)
    


    def __str__(self):
        return f"{self.film.name if self.film else self.serie.name} | {self.start_time.strftime('%H:%M:%S')}"
  
  
    # def __str__(self):
    #     return f"{self.film.name} | {self.start_time.strftime('%H:%M:%S')}"
    

def generate_bookid(length=8):
    """Hàm sinh mã BOOKID ngẫu nhiên gồm chữ và số."""
    characters = string.ascii_uppercase + string.digits  # Các ký tự chữ hoa và số
    return ''.join(random.choice(characters) for _ in range(length))


class Bookings(models.Model):
    bookid = models.CharField(max_length=8, unique=True, editable=False, blank=True)
    user = models.ForeignKey(to='user_app.User', null=True, blank=True, on_delete=models.CASCADE)
    shows = models.ForeignKey(Shows, on_delete=models.CASCADE)
    useat = models.CharField(max_length=100)
    
    @property
    def useat_as_list(self):
        return self.useat.split(',')

    def save(self, *args, **kwargs):
        # Nếu bookid chưa được tạo, tạo bookid mới
        if not self.bookid:
            self.bookid = generate_bookid()
            
            # Đảm bảo mã bookid là duy nhất
            while Bookings.objects.filter(bookid=self.bookid).exists():
                self.bookid = generate_bookid()

        super().save(*args, **kwargs)
    
    @property
    def total_price(self):
        number_of_seats = len(self.useat_as_list)
        return Decimal(number_of_seats) * self.shows.price

    @property
    def has_successful_payment(self):
        return self.book_payment.filter(vnp_ResponseCode="00").exists()

    def __str__(self):
        return f"{self.bookid} | {self.user.username} | {self.shows.film.name if self.shows.film else self.shows.serie.name} | {self.useat}"

class PaymentForm(forms.Form):
    order_id = forms.CharField(max_length=250)
    order_type = forms.CharField(max_length=20)
    amount = forms.IntegerField()
    order_desc = forms.CharField(max_length=100)
    bank_code = forms.CharField(max_length=20, required=False)
    language = forms.CharField(max_length=2)


def default_booking_id():
    default_booking = Bookings.objects.first()
    return default_booking.id if default_booking else None

class Payment(models.Model):
    bookid = models.ForeignKey(to=Bookings,on_delete=models.CASCADE,related_name='book_payment',default= default_booking_id)
    price = models.FloatField(default=0.0, null=True, blank=True)
    order_desc = models.CharField(max_length=200,null=True, blank=True)
    vnp_TransactionNo = models.CharField(max_length=200,null=True,blank=True)
    vnp_ResponseCode = models.CharField(max_length=200,null=True,blank=True)    

# Serie.objects.create(name="Default Serie", slug="default-serie")

class Comment(models.Model):
    film = models.ForeignKey(to=Film, on_delete=models.CASCADE,related_name='film_comments',null=True,blank=True)
    serie = models.ForeignKey(to=Serie, on_delete=models.CASCADE,related_name='serie_comment',null=True,blank=True)
    user = models.ForeignKey(to='user_app.User', null=True, blank=True, on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    rating = models.IntegerField(choices=[(1, '1 star'), (2, '2 stars'), (3, '3 stars'), (4, '4 stars'), (5, '5 stars')])

    status = models.CharField(max_length=20, default='approved', 
                            choices=[('approved', 'Approved'), ('pending', 'Pending'), ('rejected', 'Rejected')])



    class Meta:
        ordering = ['-created_at']  # Sắp xếp theo `created_at` giảm dần
    def __str__(self):
        if self.film:
            return f"Comment by {self.user} on Film: {self.film.name}"
        elif self.serie:
            return f"Comment by {self.user} on Serie: {self.serie.name}"
        return f"Comment by {self.user}"    
    # def __str__(self):
    #     return f"Comment by {self.user} on {self.film.name}"

class ChatHistory(models.Model):
    user = models.ForeignKey(to='user_app.User', null=True, blank=True, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)  # Thời gian tạo lịch sử chat
    last_message = models.TextField(null=True, blank=True)  # Lưu tin nhắn cuối cùng
    last_response = models.TextField(null=True, blank=True)  # Lưu phản hồi cuối cùng

    def update_last_message(self, message, response):
        self.last_message = message
        self.last_response = response
        self.save()

    def __str__(self):
        return f"Chat history for {self.user.username} created at {self.created_at}"    

class ChatMessage(models.Model):
    chat_history = models.ForeignKey(ChatHistory, on_delete=models.CASCADE, related_name="messages")  # Liên kết với lịch sử chat
    message = models.TextField()  # The user's message
    bot_response = models.TextField()  # The bot's response
    timestamp = models.DateTimeField(auto_now_add=True)  # Timestamp for when the message was created
   
    def save(self, *args, **kwargs):
        # Lưu tin nhắn
        super().save(*args, **kwargs)
        
        # Cập nhật lịch sử chat với tin nhắn và phản hồi cuối cùng
        self.chat_history.update_last_message(self.message, self.bot_response)

    def __str__(self):
        return f"Message from {self.chat_history.user.username} at {self.timestamp}"