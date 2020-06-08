from django.shortcuts import render, redirect
from django.contrib.auth.models import User, auth
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout
from binascii import hexlify
import os
'''
def register(request):
    """
    Render the new user creation page.
    """
    context = {}
    if request.POST.get('submit'):
        username = email = request.POST.get('email')
        password = request.POST.get('password')
        try:
            if '@' in email:
                api_key = hexlify(os.urandom(8))
                user = User.objects.create_user(username, email, password,
                                                last_name=api_key)
                user = authenticate(username=username, password=password)
                if user:
                    auth_login(request, user)
                    user.save()
                if request.user.is_authenticated:
                    full_path = request.get_full_path()
                    if 'next' in full_path:
                        destination = full_path.split('/')[-1]
                    else:
                        destination = 'home'
                    return redirect(destination)
            else:
                context['alert'] = 'Please enter a valid email.'

        except Exception as e:
            username_errors = [
                'UNIQUE constraint failed: auth_user.username',
                'column username is not unique'
            ]
            if str(e) in username_errors:
                # It wasn't a unique username
                context['alert'] = 'That email is already registered.'
            else:
                print(e)
    is_signed_in = request.user.is_authenticated and not request.user.is_anonymous
    context.update({"is_signed_in": is_signed_in})
    return render(request, 'register2.html', context)
'''

def login(request):
    """
    Render the login page.
    """
    context = {}
    if request.POST.get('submit'):
        username = email = request.POST.get('email')
        password = request.POST.get('password')
        try:
            user = authenticate(username=username, password=password)
            if user:
                auth_login(request, user)
                full_path = request.get_full_path()
                if 'next' in full_path:
                    print(full_path)
                    destination = full_path.split('/')[-1]
                else:
                    destination = 'home'
                return redirect(destination)
            else:
                context['alert'] = 'Username and/or password are not valid'
        except Exception as e:
            if str(e) == 'User matching query does not exist.':
                context['alert'] = 'Username and/or password are not valid'
            else:
                print(e)
    is_signed_in = request.user.is_authenticated and not request.user.is_anonymous
    context.update({"is_signed_in": is_signed_in})
    return render(request, 'login2.html', context)


def signout(request, *args,**kwargs):
    is_signed_in = False
    #logout(request)
    context = {
        'is_signed_in': is_signed_in
    }
    return render(request, 'home.html', context)

def register(request):

    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        email = request.POST['email']
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        key = hexlify(os.urandom(5)).decode()
        print(key)
        user = User.objects.create_user(password=password1, email=email, first_name=first_name,
                                        last_name=last_name, username=username, api_key=key)
        user.save()
        print('User Created')
        return redirect('/')
    else:
        return render(request, 'register2.html',)
