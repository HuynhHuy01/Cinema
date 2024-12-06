from django.db.models import Count
from django.views.generic import DetailView, ListView
from utils.tools import get_client_ip
from .models import Serie, Film, Part, Season, FilmByQuality, FilmVisit, SerieVisit, Genre, Date,Shows,Bookings,Payment,Comment,ChatHistory,ChatMessage
from django.shortcuts import get_object_or_404, render,redirect
from django.urls import reverse
from django.http import HttpResponse
from .forms import CommentForm




from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
import random,json
import numpy as np

from tensorflow import keras
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout  

import nltk
import pickle

from django.contrib.auth import logout
from django.shortcuts import redirect
# nltk.download('punkt')
# nltk.download('wordnet')

from keras.models import load_model



class SerieComponent(DetailView):
    template_name = 'serie_download_page.html'
    model = Serie
    context_object_name = 'serie'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        loaded_serie = self.object
        context['seasons'] = Season.objects.filter(serie__slug=loaded_serie.slug).all()
        context['parts'] = Part.objects.filter(season__serie__slug=loaded_serie.slug).all()
        context['trailer'] = loaded_serie.trailer.url if loaded_serie.trailer else None
        context['related_series'] = Serie.objects.filter(genre=loaded_serie.genre.first()).exclude(pk=loaded_serie.id)
        if self.request.user.is_authenticated:
            context['is_favorite'] = self.request.user.saved_series.filter(pk=loaded_serie.id).exists()
        user_ip = get_client_ip(self.request)
        user_id = None
        if self.request.user.is_authenticated:
            user_id = self.request.user.id

        has_been_visited = SerieVisit.objects.filter(ip__iexact=user_ip, serie_id=loaded_serie.id).exists()

        if not has_been_visited:
            new_visit = SerieVisit(ip=user_ip, user_id=user_id, serie_id=loaded_serie.id)
            new_visit.save()

        return context

from django.utils import timezone
from transformers import pipeline


sentiment_analyzer = pipeline("sentiment-analysis")
class FilmComponent(DetailView):
    template_name = 'film_download_page.html'
    model = Film
    context_object_name = 'film'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        loaded_film = self.object
        context['film_qualities'] = FilmByQuality.objects.prefetch_related().filter(film__slug=loaded_film.slug).all()
        context['related_films'] = Film.objects.filter(genre=loaded_film.genre.first()).exclude(pk=loaded_film.id)
        context['trailer'] = loaded_film.trailer.url if loaded_film.trailer else None

        context['comments'] = Comment.objects.filter(film=loaded_film)
        context['form'] = CommentForm()

        all_ratings = Comment.objects.filter(film=loaded_film).values_list('rating', flat=True)
        context['average_rating'] = sum(all_ratings) / len(all_ratings) if all_ratings else 0
        context['loop_time'] = range(5,0,-1)
        context['loop_times'] = range(1,6)
       
        

        if self.request.user.is_authenticated:
            context['is_favorite'] = self.request.user.saved_films.filter(pk=loaded_film.id).exists()

        user_ip = get_client_ip(self.request)
        user_id = None
        if self.request.user.is_authenticated:
            user_id = self.request.user.id

        has_been_visited = FilmVisit.objects.filter(ip__iexact=user_ip, film_id=loaded_film.id).exists()

        if not has_been_visited:
            new_visit = FilmVisit(ip=user_ip, user_id=user_id, film_id=loaded_film.id)
            new_visit.save()

        return context
    
    def post(self, request, *args, **kwargs):
        # Handle POST request for comments
        self.object = self.get_object()  # Load the film object
        
        edit_comment_id = request.POST.get('edit_comment_id')
        if edit_comment_id:
            comment = get_object_or_404(Comment, id=edit_comment_id)  # Fetch the comment
            if comment.user == request.user:  # Ensure the user is the comment owner
                comment.content = request.POST.get('content')  # Update the comment content
                # comment.content = request.POST.get('content', comment.content)
                comment.created_at = timezone.now()  # Update created_at to now
                comment.save()  # Save the comment
                return redirect(reverse('film-page', kwargs={'slug': self.object.slug}))
        
        
        # Handle comment deletion
        delete_comment_id = request.POST.get('delete_comment_id')
        if delete_comment_id:
            comment = get_object_or_404(Comment, id=delete_comment_id)  # Fetch the comment
            if comment.user == request.user:  # Ensure the user is the comment owner
                comment.delete()  # Delete the comment
                return redirect(reverse('film-page', kwargs={'slug': self.object.slug}))  

      
            
        form = CommentForm(request.POST)
        if form.is_valid():

            content = form.cleaned_data['content']
            # Perform sentiment analysis on the content
            sentiment_result = sentiment_analyzer(content)
            sentiment = sentiment_result[0]['label']
            score = sentiment_result[0]['score']

            # Default comment status is 'pending'
            status = 'pending'

        # Check sentiment score and decide the status
        if sentiment == "NEGATIVE" and score > 0.8:
            # If the comment is too negative, set status as 'rejected'
            status = 'rejected'
            return JsonResponse({'error': 'Your comment is too negative to be posted.'}, status=400)
        elif sentiment == "NEUTRAL":
            # If the comment is neutral, set status as 'pending'
            status = 'pending'
        else:
            # If it's positive, set status as 'approved'
            status = 'approved' if score > 0.5 else 'pending'


            comment = form.save(commit=False)
            comment.film = self.object
            comment.user = request.user
            comment.status = status
            comment.save()
            
            # if status == 'rejected':
            #     return JsonResponse({'error': 'Your comment has been rejected due to negative content.'}, status=400)
            
            # return redirect(reverse('film-page', kwargs={'slug': self.object.slug}))
            return JsonResponse({
            'success': 'Your comment has been posted successfully.',
            'user': comment.user.username,
            'content': comment.content,
            'rating': comment.rating
        })

    # If form is not valid, re-render page with form errors
        return JsonResponse({'error': 'There was an error submitting your comment.'}, status=400)
        
        # If form is not valid, re-render page with form errors
        context = self.get_context_data()
        context['form'] = form
        return self.render_to_response(context)


class FilmList(ListView):
    template_name = 'films_page.html'
    model = Film
    context_object_name = 'films'
    ordering = ['-imdb']
    paginate_by = 6

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        context['most_visit'] = Film.objects.filter(is_active=True).annotate(
            visit_count=Count('filmvisit')).order_by('-visit_count')[:8]
        context['all_genres'] = Genre.objects.all()
        context['all_years'] = Date.objects.all()
        # film_ids = context['films'].values_list('id', flat=True)  # Get the list of film IDs
        # context['shows'] = Shows.objects.filter(film_id__in=film_ids)

        return context


class SerieList(ListView):
    template_name = 'series_page.html'
    model = Serie
    context_object_name = 'series'
    ordering = ['-imdb']
    paginate_by = 6

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        context['most_visit'] = Serie.objects.filter(is_active=True).annotate(
            visit_count=Count('serievisit')).order_by('-visit_count')[:8]
        context['all_genres'] = Genre.objects.all()
        context['all_years'] = Date.objects.all()

        return context
        
def seat(request,id):
    show = Shows.objects.get(shows=id)
    seat = Bookings.objects.filter(shows=id)
    user_has_booking = False
    if request.user.is_authenticated:
        user_has_booking = Bookings.objects.filter(user=request.user, shows=show).exists()
    return render(request,"seat.html", {'show':show, 'seat':seat,'user_has_booking':user_has_booking})    

def ticket(request,id):
    ticket = Bookings.objects.get(id=id)
    return render(request,"ticket.html", {'ticket':ticket})

def booked(request):
    if request.method == 'POST':
        user = request.user
        seat = ','.join(request.POST.getlist('check'))
        show = request.POST['show']
        book = Bookings(useat=seat, shows_id=show, user=user)
        book.save()
        return render(request,"booked.html", {'book':book})  


def bookings(request):
    user = request.user
    book = Bookings.objects.filter(user=user.pk)
    # book_ids = book.values_list('bookid', flat=True)

    # # Use the book_ids list to filter payments
    # payments = Payment.objects.filter(book_id__in=book_ids)
    return render(request,"booking.html", {'book':book} )   



def delete(request,id):
    booked_delete = Bookings.objects.get(id=id);
    booked_delete.delete();
    return redirect('/movies/bookings')


def add_shows(request):
    return render(request,"addshow.html")


def ChatBot(request):
    return render(request,"chatbot/chat.html")


from tensorflow.keras.optimizers import SGD
def train_chatbot():
    lemmatizer = WordNetLemmatizer()

    # Load intents
    with open('jsonfile.json') as file:
        intents = json.load(file)

    words = []
    classes = []
    documents = []

    # Process intents
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ['is', 'and', 'the', 'a', 'are', 'i', 'it']]))
    classes = sorted(set(classes))

    # Save words and classes
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]

        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    # Build the model
    model = Sequential([
        Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(train_y[0]), activation='softmax')
    ])

    # Compile the model
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Train the model
    model.fit(np.array(train_x), np.array(train_y), epochs=20, batch_size=1, verbose=1)

    # Save the model
    model.save('chatbot_model.h5')
    print("Model training complete and saved as chatbot_model.h5")

train_chatbot()


def Chat(request):
   query = str(request.GET.get("query"))
   lemmatizer = WordNetLemmatizer()

   with open('jsonfile.json') as fileobj:
      readobj=json.load(fileobj)

   words = pickle.load(open('words.pkl', 'rb'))
   classes = pickle.load(open('classes.pkl', 'rb'))

   model = load_model('chatbot_model.h5')   
   
   def cleaning_up_message(message):
      message_word = word_tokenize(message)
      message_word = [lemmatizer.lemmatize(word.casefold()) for word in message_word]
      return message_word


   def bag_of_words(message):
      message_word = cleaning_up_message(message)
      bag = [0]*len(words)
      for w in message_word:
         for i, word in enumerate(words):
               if word == w:
                  bag[i] = 1
      return np.array(bag)
      
   INTENT_NOT_FOUND_THRESHOLD = 0.25

   def predict_intent_tag(message):
      BOW = bag_of_words(message)
      res = model.predict(np.array([BOW]))[0]
      results = [[i,r] for i,r in enumerate(res) if r > INTENT_NOT_FOUND_THRESHOLD ]
      results.sort(key = lambda x : x[1] , reverse = True)
      return_list = []
      for r in results:
         return_list.append({ 'intent': classes[r[0]], 'probability': str(r[1]) })
      return return_list

   def get_response(intents_list , intents_json):
      tag = intents_list[0]['intent']
      list_of_intents = intents_json['intents']
      for i in list_of_intents:
         if i['tag'] == tag :
               result= random.choice (i['responses'])
               break
      return result

   print(" Aapi is Running ! ")
   message = query
   ints = predict_intent_tag(message)
   bot_response = get_response(ints, readobj)

   user = request.user

   chat_history,created = ChatHistory.objects.get_or_create(user=user)

  
   ChatMessage.objects.create(chat_history = chat_history, message=query, bot_response=bot_response)
   return JsonResponse({"Bot": bot_response})




















import hashlib
import hmac
import json
import urllib
import urllib.parse
import urllib.request
import random
import requests
from datetime import datetime
from django.conf import settings
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render, redirect


from movie_app.models import PaymentForm
from movie_app.vnpay import vnpay


def index(request):
    return render(request, "payment/index.html", {"title": "Danh sách demo"})


def hmacsha512(key, data):
    byteKey = key.encode('utf-8')
    byteData = data.encode('utf-8')
    return hmac.new(byteKey, byteData, hashlib.sha512).hexdigest()


def payment(request, id):
    try:
        payment = Bookings.objects.get(id=id)
    except Bookings.DoesNotExist:
        return render(request, "payment/payment.html", {"title": "Thanh toán", "error": "Payment not found"})

    if request.method == 'POST':
        form = PaymentForm(request.POST)
        if form.is_valid():
            order_type = form.cleaned_data['order_type']
            order_id = form.cleaned_data['order_id']
            amount = form.cleaned_data['amount']
            order_desc = form.cleaned_data['order_desc']
            bank_code = form.cleaned_data['bank_code']
            language = form.cleaned_data['language']
            ipaddr = get_client_ip(request)

            try:
                vnp = vnpay()  # Ensure vnpay() is correctly defined
                vnp.requestData['vnp_Version'] = '2.1.0'
                vnp.requestData['vnp_Command'] = 'pay'
                vnp.requestData['vnp_TmnCode'] = settings.VNPAY_TMN_CODE
                vnp.requestData['vnp_Amount'] = amount * 100000
                vnp.requestData['vnp_CurrCode'] = 'VND'
                vnp.requestData['vnp_TxnRef'] = order_id
                vnp.requestData['vnp_OrderInfo'] = order_desc
                vnp.requestData['vnp_OrderType'] = order_type

                if language:
                    vnp.requestData['vnp_Locale'] = language
                else:
                    vnp.requestData['vnp_Locale'] = 'vn'

                if bank_code:
                    vnp.requestData['vnp_BankCode'] = bank_code

                vnp.requestData['vnp_CreateDate'] = datetime.now().strftime('%Y%m%d%H%M%S')
                vnp.requestData['vnp_IpAddr'] = ipaddr
                vnp.requestData['vnp_ReturnUrl'] = settings.VNPAY_RETURN_URL
                vnpay_payment_url = vnp.get_payment_url(settings.VNPAY_PAYMENT_URL, settings.VNPAY_HASH_SECRET_KEY)
                return redirect(vnpay_payment_url)
            except Exception as e:
                print(f"Error: {e}")
                return render(request, "payment/payment.html", {"title": "Thanh toán", "payment": payment, "error": "Payment processing failed"})
        else:
            print(form.errors)
            return render(request, "payment/payment.html", {"title": "Thanh toán", "payment": payment, "error": "Form is invalid"})
    else:
        return render(request, "payment/payment.html", {"title": "Thanh toán", "payment": payment})


def payment_ipn(request):
    inputData = request.GET
    
    if inputData:
        vnp = vnpay()
        vnp.responseData = inputData.dict()
        order_id = inputData['vnp_TxnRef']
        # order_type = inputData['vnp_OrderType']
        amount = inputData['vnp_Amount']
        order_desc = inputData['vnp_OrderInfo']
        vnp_TransactionNo = inputData['vnp_TransactionNo']
        vnp_ResponseCode = inputData['vnp_ResponseCode']
        vnp_TmnCode = inputData['vnp_TmnCode']
        vnp_PayDate = inputData['vnp_PayDate']
        vnp_BankCode = inputData['vnp_BankCode']
        vnp_CardType = inputData['vnp_CardType']
        if vnp.validate_response(settings.VNPAY_HASH_SECRET_KEY):
            # Check & Update Order Status in your Database
            # Your code here
            firstTimeUpdate = True
            totalamount = True
            if totalamount:
                if firstTimeUpdate:
                    if vnp_ResponseCode == '00':
                        print('Payment Success. Your code implement here')
                    else:
                        print('Payment Error. Your code implement here')

                    # Return VNPAY: Merchant update success
                    result = JsonResponse({'RspCode': '00', 'Message': 'Confirm Success'})
                else:
                    # Already Update
                    result = JsonResponse({'RspCode': '02', 'Message': 'Order Already Update'})
            else:
                # invalid amount
                result = JsonResponse({'RspCode': '04', 'Message': 'invalid amount'})
        else:
            # Invalid Signature
            result = JsonResponse({'RspCode': '97', 'Message': 'Invalid Signature'})
    else:
        result = JsonResponse({'RspCode': '99', 'Message': 'Invalid request'})

    return result


def payment_return(request):
    inputData = request.GET
    if inputData:
        vnp = vnpay()
        vnp.responseData = inputData.dict()
        order_id = inputData['vnp_TxnRef']
        # order_type = inputData['vnp_OrderType']
        amount = int(inputData['vnp_Amount']) / 100
        order_desc = inputData['vnp_OrderInfo']
        vnp_TransactionNo = inputData['vnp_TransactionNo']
        vnp_ResponseCode = inputData['vnp_ResponseCode']
        vnp_TmnCode = inputData['vnp_TmnCode']
        vnp_PayDate = inputData['vnp_PayDate']
        vnp_BankCode = inputData['vnp_BankCode']
        vnp_CardType = inputData['vnp_CardType']

        try:
           booking_instance = Bookings.objects.get(bookid=order_id)
        except Bookings.DoesNotExist:
            return HttpResponse("Booking not found", status=404)

        payment = Payment.objects.create(
            bookid = booking_instance,
            price = amount,
            order_desc = order_desc,
            vnp_TransactionNo = vnp_TransactionNo,
            vnp_ResponseCode = vnp_ResponseCode
        )



        if vnp.validate_response(settings.VNPAY_HASH_SECRET_KEY):
            if vnp_ResponseCode == "00":
                return render(request, "payment/payment_return.html", {"title": "Kết quả thanh toán",
                                                               "result": "Thành công", "order_id": order_id,
                                                               "amount": amount,
                                                               "order_desc": order_desc,
                                                               "vnp_TransactionNo": vnp_TransactionNo,
                                                               "vnp_ResponseCode": vnp_ResponseCode})
            else:
                return render(request, "payment/payment_return.html", {"title": "Kết quả thanh toán",
                                                               "result": "Lỗi", "order_id": order_id,
                                                               "amount": amount,
                                                               "order_desc": order_desc,
                                                               "vnp_TransactionNo": vnp_TransactionNo,
                                                               "vnp_ResponseCode": vnp_ResponseCode})
        else:
            return render(request, "payment/payment_return.html",
                          {"title": "Kết quả thanh toán", "result": "Lỗi", "order_id": order_id, "amount": amount,
                           "order_desc": order_desc, "vnp_TransactionNo": vnp_TransactionNo,
                           "vnp_ResponseCode": vnp_ResponseCode, "msg": "Sai checksum"})
    else:
        return render(request, "payment/payment_return.html", {"title": "Kết quả thanh toán", "result": ""})


def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

n = random.randint(10**11, 10**12 - 1)
n_str = str(n)
while len(n_str) < 12:
    n_str = '0' + n_str


def query(request):
    if request.method == 'GET':
        return render(request, "payment/query.html", {"title": "Kiểm tra kết quả giao dịch"})

    url = settings.VNPAY_API_URL
    secret_key = settings.VNPAY_HASH_SECRET_KEY
    vnp_TmnCode = settings.VNPAY_TMN_CODE
    vnp_Version = '2.1.0'

    vnp_RequestId = n_str
    vnp_Command = 'querydr'
    vnp_TxnRef = request.POST['order_id']
    vnp_OrderInfo = 'kiem tra gd'
    vnp_TransactionDate = request.POST['trans_date']
    vnp_CreateDate = datetime.now().strftime('%Y%m%d%H%M%S')
    vnp_IpAddr = get_client_ip(request)

    hash_data = "|".join([
        vnp_RequestId, vnp_Version, vnp_Command, vnp_TmnCode,
        vnp_TxnRef, vnp_TransactionDate, vnp_CreateDate,
        vnp_IpAddr, vnp_OrderInfo
    ])

    secure_hash = hmac.new(secret_key.encode(), hash_data.encode(), hashlib.sha512).hexdigest()

    data = {
        "vnp_RequestId": vnp_RequestId,
        "vnp_TmnCode": vnp_TmnCode,
        "vnp_Command": vnp_Command,
        "vnp_TxnRef": vnp_TxnRef,
        "vnp_OrderInfo": vnp_OrderInfo,
        "vnp_TransactionDate": vnp_TransactionDate,
        "vnp_CreateDate": vnp_CreateDate,
        "vnp_IpAddr": vnp_IpAddr,
        "vnp_Version": vnp_Version,
        "vnp_SecureHash": secure_hash
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_json = json.loads(response.text)
    else:
        response_json = {"error": f"Request failed with status code: {response.status_code}"}

    return render(request, "payment/query.html", {"title": "Kiểm tra kết quả giao dịch", "response_json": response_json})

def refund(request):
    if request.method == 'GET':
        return render(request, "payment/refund.html", {"title": "Hoàn tiền giao dịch"})

    url = settings.VNPAY_API_URL
    secret_key = settings.VNPAY_HASH_SECRET_KEY
    vnp_TmnCode = settings.VNPAY_TMN_CODE
    vnp_RequestId = n_str
    vnp_Version = '2.1.0'
    vnp_Command = 'refund'
    vnp_TransactionType = request.POST['TransactionType']
    vnp_TxnRef = request.POST['order_id']
    vnp_Amount = request.POST['amount']
    vnp_OrderInfo = request.POST['order_desc']
    vnp_TransactionNo = '0'
    vnp_TransactionDate = request.POST['trans_date']
    vnp_CreateDate = datetime.now().strftime('%Y%m%d%H%M%S')
    vnp_CreateBy = 'user01'
    vnp_IpAddr = get_client_ip(request)

    hash_data = "|".join([
        vnp_RequestId, vnp_Version, vnp_Command, vnp_TmnCode, vnp_TransactionType, vnp_TxnRef,
        vnp_Amount, vnp_TransactionNo, vnp_TransactionDate, vnp_CreateBy, vnp_CreateDate,
        vnp_IpAddr, vnp_OrderInfo
    ])

    secure_hash = hmac.new(secret_key.encode(), hash_data.encode(), hashlib.sha512).hexdigest()

    data = {
        "vnp_RequestId": vnp_RequestId,
        "vnp_TmnCode": vnp_TmnCode,
        "vnp_Command": vnp_Command,
        "vnp_TxnRef": vnp_TxnRef,
        "vnp_Amount": vnp_Amount,
        "vnp_OrderInfo": vnp_OrderInfo,
        "vnp_TransactionDate": vnp_TransactionDate,
        "vnp_CreateDate": vnp_CreateDate,
        "vnp_IpAddr": vnp_IpAddr,
        "vnp_TransactionType": vnp_TransactionType,
        "vnp_TransactionNo": vnp_TransactionNo,
        "vnp_CreateBy": vnp_CreateBy,
        "vnp_Version": vnp_Version,
        "vnp_SecureHash": secure_hash
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_json = json.loads(response.text)
    else:
        response_json = {"error": f"Request failed with status code: {response.status_code}"}

    return render(request, "payment/refund.html", {"title": "Kết quả hoàn tiền giao dịch", "response_json": response_json})

