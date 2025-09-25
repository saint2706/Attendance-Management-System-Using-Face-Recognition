"""Views for the recognition app."""

from __future__ import annotations

import datetime
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Dict

import cv2
import dlib
import face_recognition
import imutils
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, QuerySet
from django.shortcuts import redirect, render
from django.utils import timezone
from django_pandas.io import read_frame
from face_recognition.face_recognition_cli import image_files_in_folder
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.video import VideoStream
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .forms import DateForm, DateForm_2, UsernameAndDateForm, usernameForm
from users.models import Present, Time


mpl.use('Agg')


#utility functions:
def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	
	return False

def create_dataset(username):
	id = username
	if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(id))==False):
		os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
	directory='face_recognition_data/training_dataset/{}/'.format(id)

	# Detect face
	#Loading the HOG face detector and the shape predictpr for allignment

	print("[INFO] Loading the facial detector")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	#capture images from the webcam and process and detect the face
	# Initialize the video stream
	print("[INFO] Initializing Video stream")
	vs = VideoStream(src=0).start()
	#time.sleep(2.0) ####CHECK######

	# Our identifier
	# We will put the id here and we will store the id with a face, so that later we can identify whose face it is
	
	# Our dataset naming counter
	sampleNum = 0
	# Capturing the faces one by one and detect the faces and showing it on the window
	while(True):
		# Capturing the image
		#vs.read each frame
		frame = vs.read()
		#Resize each image
		frame = imutils.resize(frame ,width = 800)
		#the returned img is a colored image but for the classifier to work we need a greyscale image
		#to convert
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#To store the faces
		#This will detect all the images in the current frame, and it will return the coordinates of the faces
		#Takes in image and some other parameter for accurate result
		faces = detector(gray_frame,0)
		#In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
		
		
			


		for face in faces:
			print("inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)
			# Whenever the program captures the face, we will write that is a folder
			# Before capturing the face, we need to tell the script whose face it is
			# For that we will need an identifier, here we call it id
			# So now we captured a face, we need to write it in a file
			sampleNum = sampleNum+1
			# Saving the image dataset, but only the face part, cropping the rest
			
			if face is None:
				print("face is none")
				continue


			

			cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg'	, face_aligned)
			face_aligned = imutils.resize(face_aligned ,width = 400)
			#cv2.imshow("Image Captured",face_aligned)
			# @params the initial point of the rectangle will be x,y and
			# @params end point will be x+width and y+height
			# @params along with color of the rectangle
			# @params thickness of the rectangle
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Add Images",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		cv2.waitKey(1)
		#To get out of the loop
		if(sampleNum>300):
			break
	
	#Stoping the videostream
	vs.stop()
	# destroying all the windows
	cv2.destroyAllWindows()


def predict(face_aligned,svc,threshold=0.7):
        face_encodings=np.zeros((1,128))
        try:
                x_face_locations=face_recognition.face_locations(face_aligned)
                faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)
                if(len(faces_encodings)==0):
                        return (-1,0.0)

        except:

                return (-1,0.0)

        prob=svc.predict_proba(faces_encodings)
        best_index=int(np.argmax(prob[0]))
        best_prob=float(prob[0][best_index])
        if(best_prob<=threshold):
                return (-1,0.0)

        return (best_index,best_prob)


def vizualize_Data(embedded, targets,):
	
	X_embedded = TSNE(n_components=2).fit_transform(embedded)

	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

	plt.legend(bbox_to_anchor=(1, 1));
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()	
	plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
	plt.close()



def update_attendance_in_db_in(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		try:
		   qs=Present.objects.get(user=user,date=today)
		except :
			qs= None
		
		if qs is None:
			if present[person]==True:
						a=Present(user=user,date=today,present=True)
						a.save()
			else:
				a=Present(user=user,date=today,present=False)
				a.save()
		else:
			if present[person]==True:
				qs.present=True
				qs.save(update_fields=['present'])
		if present[person]==True:
			a=Time(user=user,date=today,time=time, out=False)
			a.save()


			
		




def update_attendance_in_db_out(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		if present[person]==True:
			a=Time(user=user,date=today,time=time, out=True)
			a.save()
		
=======
mpl.use("Agg")



logger = logging.getLogger(__name__)


DATA_ROOT = Path(settings.BASE_DIR) / "face_recognition_data"
TRAINING_DATASET_ROOT = DATA_ROOT / "training_dataset"
SHAPE_PREDICTOR_PATH = DATA_ROOT / "shape_predictor_68_face_landmarks.dat"
SVC_MODEL_PATH = DATA_ROOT / "svc.sav"
CLASSES_PATH = DATA_ROOT / "classes.npy"

APP_STATIC_ROOT = Path(__file__).resolve().parent / "static" / "recognition" / "img"
TRAINING_VIZ_PATH = APP_STATIC_ROOT / "training_visualisation.png"
HOURS_VS_DATE_PATH = APP_STATIC_ROOT / "attendance_graphs" / "hours_vs_date" / "1.png"
EMPLOYEE_LOGIN_PATH = APP_STATIC_ROOT / "attendance_graphs" / "employee_login" / "1.png"
HOURS_VS_EMPLOYEE_PATH = (
    APP_STATIC_ROOT / "attendance_graphs" / "hours_vs_employee" / "1.png"
)
THIS_WEEK_PATH = APP_STATIC_ROOT / "attendance_graphs" / "this_week" / "1.png"
LAST_WEEK_PATH = APP_STATIC_ROOT / "attendance_graphs" / "last_week" / "1.png"


def _ensure_directory(path: Path) -> None:
    """Ensure the parent directory for the given path exists."""

    path.parent.mkdir(parents=True, exist_ok=True)


def load_recognition_artifacts():
    """Load the persisted recognition artifacts."""

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))
    with open(SVC_MODEL_PATH, "rb") as model_file:
        svc = pickle.load(model_file)
    face_aligner = FaceAligner(predictor, desiredFaceWidth=96)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(str(CLASSES_PATH))
    return detector, face_aligner, svc, encoder


def username_present(username: str) -> bool:
    """Return whether the given username exists in the system."""

    return User.objects.filter(username=username).exists()


def create_dataset(username: str) -> None:
    """Capture and store face images for the provided username."""

    dataset_directory = TRAINING_DATASET_ROOT / username
    dataset_directory.mkdir(parents=True, exist_ok=True)

    logger.info("Loading the facial detector")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))
    face_aligner = FaceAligner(predictor, desiredFaceWidth=96)

    logger.info("Initializing video stream")
    video_stream = VideoStream(src=0).start()

    sample_number = 0
    while True:
        frame = video_stream.read()
        frame = imutils.resize(frame, width=800)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 0)

        for face in faces:
            logger.debug("Processing detected face for user %s", username)

            if face is None:
                logger.debug("Skipping empty face detection for user %s", username)
                continue

            face_aligned = face_aligner.align(frame, gray_frame, face)
            sample_number += 1
            output_path = dataset_directory / f"{sample_number}.jpg"
            cv2.imwrite(str(output_path), face_aligned)
            imutils.resize(face_aligned, width=400)
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.waitKey(50)

        cv2.imshow("Add Images", frame)
        cv2.waitKey(1)
        if sample_number > 300:
            break

    video_stream.stop()
    cv2.destroyAllWindows()


def predict(face_aligned, svc, threshold: float = 0.7):
    """Predict the identity of the aligned face using the trained classifier."""

    try:
        x_face_locations = face_recognition.face_locations(face_aligned)
        faces_encodings = face_recognition.face_encodings(
            face_aligned, known_face_locations=x_face_locations
        )
        if len(faces_encodings) == 0:
            return (-1, 0.0)

    except Exception:

        logger.exception("Failed to encode face for prediction")
        return (-1, 0.0)

    prob = svc.predict_proba(faces_encodings)
    best_index = int(np.argmax(prob[0]))
    best_prob = float(prob[0][best_index])
    if best_prob <= threshold:
        return (-1, best_prob)

    return (best_index, best_prob)


def vizualize_Data(embedded, targets) -> None:
    """Visualise the embedded face encodings to understand clustering."""

    X_embedded = TSNE(n_components=2).fit_transform(embedded)

    for target in set(targets):
        idx = targets == target
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=target)

    plt.legend(bbox_to_anchor=(1, 1))
    rcParams.update({"figure.autolayout": True})
    plt.tight_layout()
    _ensure_directory(TRAINING_VIZ_PATH)
    plt.savefig(TRAINING_VIZ_PATH)
    plt.close()


def update_attendance_in_db_in(present: Dict[str, bool]) -> None:
    """Persist check-in attendance information for the provided users."""

    today = timezone.localdate()
    current_time = timezone.now()
    for person, is_present in present.items():
        user = User.objects.get(username=person)
        qs = Present.objects.filter(user=user, date=today).first()

        if qs is None:
            attendance = Present(user=user, date=today, present=is_present)
            attendance.save()
        elif is_present and not qs.present:
            qs.present = True
            qs.save(update_fields=["present"])

        if is_present:
            Time.objects.create(user=user, date=today, time=current_time, out=False)


def update_attendance_in_db_out(present: Dict[str, bool]) -> None:
    """Persist check-out attendance information for the provided users."""

    today = timezone.localdate()
    current_time = timezone.now()
    for person, is_present in present.items():
        if not is_present:
            continue

        user = User.objects.get(username=person)
        Time.objects.create(user=user, date=today, time=current_time, out=True)


def check_validity_times(times_all):

    if len(times_all) > 0:
        sign = times_all.first().out
    else:
        sign = True
    times_in = times_all.filter(out=False)
    times_out = times_all.filter(out=True)
    if len(times_in) != len(times_out):
        sign = True
    break_hourss = 0
    if sign == True:
        check = False
        break_hourss = 0
        return (check, break_hourss)
    prev = True
    prev_time = times_all.first().time

    for obj in times_all:
        curr = obj.out
        if curr == prev:
            check = False
            break_hourss = 0
            return (check, break_hourss)
        if curr == False:
            curr_time = obj.time
            to = curr_time
            ti = prev_time
            break_time = ((to - ti).total_seconds()) / 3600
            break_hourss += break_time

        else:
            prev_time = obj.time

        prev = curr

    return (True, break_hourss)


def convert_hours_to_hours_mins(hours):

    h = int(hours)
    hours -= h
    m = hours * 60
    m = math.ceil(m)
    return str(str(h) + " hrs " + str(m) + "  mins")


# used
def hours_vs_date_given_employee(present_qs, time_qs, admin=True):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    qs = present_qs

    for obj in qs:
        date = obj.date
        times_in = time_qs.filter(date=date, out=False).order_by("time")
        times_out = time_qs.filter(date=date, out=True).order_by("time")
        times_all = time_qs.filter(date=date).order_by("time")
        obj.time_in = times_in.first().time if times_in else None
        obj.time_out = times_out.last().time if times_out else None
        if obj.time_in and obj.time_out:
            hours = (obj.time_out - obj.time_in).total_seconds() / 3600
            obj.hours = hours
        else:
            obj.hours = 0
        check, break_hours = check_validity_times(times_all)
        obj.break_hours = break_hours if check else 0
        df_hours.append(obj.hours)
        df_break_hours.append(obj.break_hours)
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)

    df = read_frame(qs)
    df["hours"] = df_hours
    df["break_hours"] = df_break_hours
    logger.debug("Attendance dataframe: %s", df)

    sns.barplot(data=df, x="date", y="hours")
    plt.xticks(rotation="vertical")
    rcParams.update({"figure.autolayout": True})
    plt.tight_layout()
    target_path = HOURS_VS_DATE_PATH if admin else EMPLOYEE_LOGIN_PATH
    _ensure_directory(target_path)
    plt.savefig(target_path)
    plt.close()
    return qs


# used
def hours_vs_employee_given_date(present_qs, time_qs):
    register_matplotlib_converters()
    df_hours = []
    df_break_hours = []
    df_username = []
    qs = present_qs

    for obj in qs:
        user = obj.user
        times_in = time_qs.filter(user=user, out=False)
        times_out = time_qs.filter(user=user, out=True)
        times_all = time_qs.filter(user=user)
        obj.time_in = times_in.first().time if times_in else None
        obj.time_out = times_out.last().time if times_out else None
        if obj.time_in and obj.time_out:
            obj.hours = (obj.time_out - obj.time_in).total_seconds() / 3600
        else:
            obj.hours = 0
        check, break_hours = check_validity_times(times_all)
        obj.break_hours = break_hours if check else 0
        df_hours.append(obj.hours)
        df_username.append(user.username)
        df_break_hours.append(obj.break_hours)
        obj.hours = convert_hours_to_hours_mins(obj.hours)
        obj.break_hours = convert_hours_to_hours_mins(obj.break_hours)

    df = read_frame(qs)
    df["hours"] = df_hours
    df["username"] = df_username
    df["break_hours"] = df_break_hours

    sns.barplot(data=df, x="username", y="hours")
    plt.xticks(rotation="vertical")
    rcParams.update({"figure.autolayout": True})
    plt.tight_layout()
    _ensure_directory(HOURS_VS_EMPLOYEE_PATH)
    plt.savefig(HOURS_VS_EMPLOYEE_PATH)
    plt.close()
    return qs


def total_number_employees() -> int:
    qs = User.objects.all()
    return len(qs) - 1


def employees_present_today() -> int:
    today = timezone.localdate()
    qs = Present.objects.filter(date=today, present=True)
    return len(qs)


# used
def this_week_emp_count_vs_date() -> None:
    today = timezone.localdate()
    some_day_last_week = today - datetime.timedelta(days=7)
    monday_of_last_week = some_day_last_week - datetime.timedelta(
        days=(some_day_last_week.isocalendar()[2] - 1)
    )
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    qs = Present.objects.filter(date__gte=monday_of_this_week, date__lte=today)
    str_dates = []
    emp_count = []
    str_dates_all = []
    emp_cnt_all = []
    cnt = 0

    for obj in qs:
        date = obj.date
        str_dates.append(str(date))
        qs = Present.objects.filter(date=date, present=True)
        emp_count.append(len(qs))

    while cnt < 5:
        date = str(monday_of_this_week + datetime.timedelta(days=cnt))
        cnt += 1
        str_dates_all.append(date)
        if str_dates.count(date) > 0:
            idx = str_dates.index(date)

            emp_cnt_all.append(emp_count[idx])
        else:
            emp_cnt_all.append(0)

    df = pd.DataFrame()
    df["date"] = str_dates_all
    df["Number of employees"] = emp_cnt_all

    sns.lineplot(data=df, x="date", y="Number of employees")
    _ensure_directory(THIS_WEEK_PATH)
    plt.savefig(THIS_WEEK_PATH)
    plt.close()


# used
def last_week_emp_count_vs_date() -> None:
    today = timezone.localdate()
    some_day_last_week = today - datetime.timedelta(days=7)
    monday_of_last_week = some_day_last_week - datetime.timedelta(
        days=(some_day_last_week.isocalendar()[2] - 1)
    )
    monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
    qs = Present.objects.filter(
        date__gte=monday_of_last_week, date__lt=monday_of_this_week
    )
    str_dates = []
    emp_count = []

    str_dates_all = []
    emp_cnt_all = []
    cnt = 0

    for obj in qs:
        date = obj.date
        str_dates.append(str(date))
        qs = Present.objects.filter(date=date, present=True)
        emp_count.append(len(qs))

    while cnt < 5:
        date = str(monday_of_last_week + datetime.timedelta(days=cnt))
        cnt += 1
        str_dates_all.append(date)
        if str_dates.count(date) > 0:
            idx = str_dates.index(date)

            emp_cnt_all.append(emp_count[idx])

        else:
            emp_cnt_all.append(0)

    df = pd.DataFrame()
    df["date"] = str_dates_all
    df["emp_count"] = emp_cnt_all

    sns.lineplot(data=df, x="date", y="emp_count")
    _ensure_directory(LAST_WEEK_PATH)
    plt.savefig(LAST_WEEK_PATH)
    plt.close()


# Create your views here.
def home(request):

    return render(request, "recognition/home.html")


@login_required
def dashboard(request):
    if request.user.username == "admin":
        logger.debug("Rendering admin dashboard for %s", request.user)
        return render(request, "recognition/admin_dashboard.html")
    else:
        logger.debug("Rendering employee dashboard for %s", request.user)

        return render(request, "recognition/employee_dashboard.html")


@login_required
def add_photos(request):
    if request.user.username != "admin":
        return redirect("not-authorised")
    if request.method == "POST":
        form = usernameForm(request.POST)
        data = request.POST.copy()
        username = data.get("username")
        if username_present(username):
            create_dataset(username)
            messages.success(request, f"Dataset Created")
            return redirect("add-photos")
        else:
            messages.warning(
                request, f"No such username found. Please register employee first."
            )
            return redirect("dashboard")

    else:

        form = usernameForm()
        return render(request, "recognition/add_photos.html", {"form": form})


def mark_your_attendance(request):

	
	

	
	detector = dlib.get_frontal_face_detector()
	
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	svc_save_path="face_recognition_data/svc.sav"	


		
			
	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')


	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False

	

	vs = VideoStream(src=0).start()
	
	sampleNum = 0
	
	while(True):
		
		frame = vs.read()
		
		frame = imutils.resize(frame ,width = 800)
		
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		faces = detector(gray_frame,0)
		
		


		for face in faces:
			print("INFO : inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
					
			
                        (pred,prob)=predict(face_aligned,svc)



                        if(pred!=-1):

                                person_name=encoder.inverse_transform([pred])[0]
                                pred=person_name
                                if count[pred] == 0:
                                        start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1

				if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
					 count[pred] = 0
				else:
				#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
                                        log_time[pred] = datetime.datetime.now()
                                        count[pred] = count.get(pred,0) + 1
                                        print(pred, present[pred], count[pred])
                                cv2.putText(frame, f"{person_name} {prob:.2f}", (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			
			
			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Mark Attendance - In - Press q to exit",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		#cv2.waitKey(1)
		#To get out of the loop
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	
	#Stoping the videostream
	vs.stop()

	# destroying all the windows
	cv2.destroyAllWindows()
	update_attendance_in_db_in(present)
	return redirect('home')



def mark_your_attendance_out(request):
	

	
	detector = dlib.get_frontal_face_detector()
	
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	svc_save_path="face_recognition_data/svc.sav"	


		
			
	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')


	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False

	

	vs = VideoStream(src=0).start()
	
	sampleNum = 0
	
	while(True):
		
		frame = vs.read()
		
		frame = imutils.resize(frame ,width = 800)
		
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		faces = detector(gray_frame,0)
		
		


		for face in faces:
			print("INFO : inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
					
			
                        (pred,prob)=predict(face_aligned,svc)



                        if(pred!=-1):

                                person_name=encoder.inverse_transform([pred])[0]
                                pred=person_name
                                if count[pred] == 0:
                                        start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1

				if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
					 count[pred] = 0
				else:
				#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
                                        log_time[pred] = datetime.datetime.now()
                                        count[pred] = count.get(pred,0) + 1
                                        print(pred, present[pred], count[pred])
                                cv2.putText(frame, f"{person_name} {prob:.2f}", (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			
			
			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Mark Attendance- Out - Press q to exit",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		#cv2.waitKey(1)
		#To get out of the loop
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	
	#Stoping the videostream
	vs.stop()

	# destroying all the windows
	cv2.destroyAllWindows()
	update_attendance_in_db_out(present)
	return redirect('home')


=======
    detector, fa, svc, encoder = load_recognition_artifacts()

    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    labels = [encoder.inverse_transform([i])[0] for i in range(no_of_faces)]
    count = {label: 0 for label in labels}
    present = {label: False for label in labels}
    log_time: Dict[str, datetime.datetime] = {}
    start: Dict[str, float] = {}

    vs = VideoStream(src=0).start()

    try:
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=800)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame, 0)

            for face in faces:
                logger.debug("Processing face for attendance check-in")
                (x, y, w, h) = face_utils.rect_to_bb(face)

                face_aligned = fa.align(frame, gray_frame, face)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

                pred, prob = predict(face_aligned, svc)

                if pred != -1:
                    person_name = encoder.inverse_transform([pred])[0]
                    if count[person_name] == 0:
                        start[person_name] = time.time()
                    if (
                        count[person_name] == 4
                        and (time.time() - start[person_name]) > 1.2
                    ):
                        count[person_name] = 0
                    else:
                        present[person_name] = True
                        log_time[person_name] = timezone.now()
                        count[person_name] = count.get(person_name, 0) + 1
                        logger.debug(
                            "Marked %s present with count %s",
                            person_name,
                            count[person_name],
                        )
                    cv2.putText(
                        frame,
                        f"{person_name} {prob:.2f}",
                        (x + 6, y + h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                else:
                    person_name = "unknown"
                    cv2.putText(
                        frame,
                        person_name,
                        (x + 6, y + h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            cv2.imshow("Mark Attendance - In - Press q to exit", frame)
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                break
    finally:
        vs.stop()
        cv2.destroyAllWindows()

    update_attendance_in_db_in(present)
    return redirect("home")


def mark_your_attendance_out(request):
    detector, fa, svc, encoder = load_recognition_artifacts()

    faces_encodings = np.zeros((1, 128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    labels = [encoder.inverse_transform([i])[0] for i in range(no_of_faces)]
    count = {label: 0 for label in labels}
    present = {label: False for label in labels}
    log_time: Dict[str, datetime.datetime] = {}
    start: Dict[str, float] = {}

    vs = VideoStream(src=0).start()

    try:
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=800)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame, 0)

            for face in faces:
                logger.debug("Processing face for attendance check-out")
                (x, y, w, h) = face_utils.rect_to_bb(face)

                face_aligned = fa.align(frame, gray_frame, face)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

                pred, prob = predict(face_aligned, svc)

                if pred != -1:
                    person_name = encoder.inverse_transform([pred])[0]
                    if count[person_name] == 0:
                        start[person_name] = time.time()
                    if (
                        count[person_name] == 4
                        and (time.time() - start[person_name]) > 1.5
                    ):
                        count[person_name] = 0
                    else:
                        present[person_name] = True
                        log_time[person_name] = timezone.now()
                        count[person_name] = count.get(person_name, 0) + 1
                        logger.debug(
                            "Marked %s for check-out with count %s",
                            person_name,
                            count[person_name],
                        )
                    cv2.putText(
                        frame,
                        f"{person_name} {prob:.2f}",
                        (x + 6, y + h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                else:
                    person_name = "unknown"
                    cv2.putText(
                        frame,
                        person_name,
                        (x + 6, y + h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            cv2.imshow("Mark Attendance- Out - Press q to exit", frame)
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                break
    finally:
        vs.stop()
        cv2.destroyAllWindows()

    update_attendance_in_db_out(present)
    return redirect("home")



@login_required
def train(request):
    if request.user.username != "admin":
        return redirect("not-authorised")

    training_dir = TRAINING_DATASET_ROOT
    if not training_dir.exists():
        messages.warning(
            request,
            "No training data available. Please add images before training the model.",
        )
        return redirect("dashboard")

    feature_vectors = []
    labels = []

    for person_dir in sorted(training_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        for image_path_str in image_files_in_folder(str(person_dir)):
            image_path = Path(image_path_str)
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Skipping unreadable training image %s", image_path)
                try:
                    image_path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Failed to remove unreadable image %s", image_path)
                continue
            try:
                encoding = face_recognition.face_encodings(image)[0]
            except IndexError:
                logger.warning("No face found in %s; removing file", image_path)
                try:
                    image_path.unlink(missing_ok=True)
                except OSError:
                    logger.exception(
                        "Failed to remove image without faces %s", image_path
                    )
                continue

            feature_vectors.append(encoding.tolist())
            labels.append(person_dir.name)

    if not feature_vectors:
        messages.warning(request, "No valid training images were found.")
        return redirect("dashboard")

    targets = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    X1 = np.array(feature_vectors)

    _ensure_directory(CLASSES_PATH)
    np.save(str(CLASSES_PATH), encoder.classes_)

    svc = SVC(kernel="linear", probability=True)
    svc.fit(X1, encoded_labels)

    _ensure_directory(SVC_MODEL_PATH)
    with open(SVC_MODEL_PATH, "wb") as model_file:
        pickle.dump(svc, model_file)

    vizualize_Data(X1, targets)

    messages.success(request, "Training Complete.")

    return render(request, "recognition/train.html")


@login_required
def not_authorised(request):
    return render(request, "recognition/not_authorised.html")


@login_required
def view_attendance_home(request):
    total_num_of_emp = total_number_employees()
    emp_present_today = employees_present_today()
    this_week_emp_count_vs_date()
    last_week_emp_count_vs_date()
    return render(
        request,
        "recognition/view_attendance_home.html",
        {"total_num_of_emp": total_num_of_emp, "emp_present_today": emp_present_today},
    )


@login_required
def view_attendance_date(request):
    if request.user.username != "admin":
        return redirect("not-authorised")
    qs = None
    time_qs = None
    present_qs = None

    if request.method == "POST":
        form = DateForm(request.POST)
        if form.is_valid():
            date = form.cleaned_data.get("date")
            logger.debug("Viewing attendance for date %s", date)
            time_qs = Time.objects.filter(date=date)
            present_qs = Present.objects.filter(date=date)
            if time_qs or present_qs:
                qs = hours_vs_employee_given_date(present_qs, time_qs)
                return render(
                    request,
                    "recognition/view_attendance_date.html",
                    {"form": form, "qs": qs},
                )
            messages.warning(request, "No records for selected date.")
            return redirect("view-attendance-date")
    else:
        form = DateForm()
    return render(
        request, "recognition/view_attendance_date.html", {"form": form, "qs": qs}
    )


@login_required
def view_attendance_employee(request):
    if request.user.username != "admin":
        return redirect("not-authorised")
    time_qs = None
    present_qs = None
    qs = None

    if request.method == "POST":
        form = UsernameAndDateForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            if username_present(username):
                user = User.objects.get(username=username)
                time_qs = Time.objects.filter(user=user)
                present_qs = Present.objects.filter(user=user)
                date_from = form.cleaned_data.get("date_from")
                date_to = form.cleaned_data.get("date_to")
                if date_to < date_from:
                    messages.warning(request, "Invalid date selection.")
                    return redirect("view-attendance-employee")
                time_qs = time_qs.filter(
                    date__gte=date_from, date__lte=date_to
                ).order_by("-date")
                present_qs = present_qs.filter(
                    date__gte=date_from, date__lte=date_to
                ).order_by("-date")
                if time_qs or present_qs:
                    qs = hours_vs_date_given_employee(present_qs, time_qs, admin=True)
                    return render(
                        request,
                        "recognition/view_attendance_employee.html",
                        {"form": form, "qs": qs},
                    )
                messages.warning(request, "No records for selected duration.")
                return redirect("view-attendance-employee")
            messages.warning(request, "No such username found.")
            return redirect("view-attendance-employee")
    else:
        form = UsernameAndDateForm()
    return render(
        request, "recognition/view_attendance_employee.html", {"form": form, "qs": qs}
    )


def view_my_attendance_employee_login(request):
    if request.user.username == "admin":
        return redirect("not-authorised")
    qs = None
    time_qs = None
    present_qs = None
    if request.method == "POST":
        form = DateForm_2(request.POST)
        if form.is_valid():
            u = request.user
            time_qs = Time.objects.filter(user=u)
            present_qs = Present.objects.filter(user=u)
            date_from = form.cleaned_data.get("date_from")
            date_to = form.cleaned_data.get("date_to")
            if date_to < date_from:
                messages.warning(request, f"Invalid date selection.")
                return redirect("view-my-attendance-employee-login")
            else:

                time_qs = (
                    time_qs.filter(date__gte=date_from)
                    .filter(date__lte=date_to)
                    .order_by("-date")
                )
                present_qs = (
                    present_qs.filter(date__gte=date_from)
                    .filter(date__lte=date_to)
                    .order_by("-date")
                )

                if len(time_qs) > 0 or len(present_qs) > 0:
                    qs = hours_vs_date_given_employee(present_qs, time_qs, admin=False)
                    return render(
                        request,
                        "recognition/view_my_attendance_employee_login.html",
                        {"form": form, "qs": qs},
                    )
                else:

                    messages.warning(request, f"No records for selected duration.")
                    return redirect("view-my-attendance-employee-login")
    else:

        form = DateForm_2()
        return render(
            request,
            "recognition/view_my_attendance_employee_login.html",
            {"form": form, "qs": qs},
        )
