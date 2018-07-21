"""
    Face Recognition
    =================
    
    a) Install face_recognition python package
    		
    	$sudo pip3 install face_recognition
    	
    b) Install OpenCV  (MacOS)
    
    	$brew install opencv3 --with-ffmpeg -v (Python 2.7)
    	$brew install opencv3 --with-python3 --with-ffmpeg -v (Python 3.6)
    	
    
    b) Create "faces" folder and add photos to it.
    
    c) Usage:
    
    	$python3 face.py
    
"""

import face_recognition
import cv2
import os
import time

video_capture = cv2.VideoCapture(0)
#length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

video_capture.set(3,500)
video_capture.set(4,400)

time.sleep(2)
video_capture.set(15, -8.0)


WINDOW_NAME = 'Face Recognition'
cv2.namedWindow(WINDOW_NAME,  cv2.WINDOW_AUTOSIZE)


"""
 Enable the below code to record and save in video file format 
"""

# Create an output movie file (make sure resolution/frame rate matches input video!)

shouldRecord = True				# Make it "True" to record and save

if shouldRecord == True:

	screen_width = int(video_capture.get(3))
	screen_height = int(video_capture.get(4))

	# print("screen_width : ", screen_width)
	# print("screen_height : ", screen_height)

	resolution = (screen_width, screen_height)
	frameRate = 10.0

	# fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# fourcc2 = cv2.VideoWriter_fourcc(*'MP4V')
	fourcc3 = cv2.VideoWriter_fourcc(*'H264')


	# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
	out = cv2.VideoWriter('output.mp4',fourcc3, frameRate, resolution)





#---------------------------------------------------------

known_faces = []
known_face_names = []


"""
	1. Create a folder called "faces" and put the image files inside it.
	2. Name the images as PersonName.jpg or PersonName.png
"""

files = os.listdir( os.path.join(os.getcwd(), "faces") )
# print(files)

for file_name in files:
    
    file_path = os.path.join(os.getcwd(), "faces", file_name)
    #print("file_path : " + file_path)
    
    photo = face_recognition.load_image_file(file_path)
    photo_encoding = face_recognition.face_encodings(photo)[0]
    person_name = file_name.split(".")[0].capitalize()

    known_faces.append(photo_encoding)
    known_face_names.append(person_name)

    print(str(files.index(file_name) + 1) + "." + person_name)


# print("\nknown_face_names : " + str(known_face_names))


#--------




# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break


    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # print("match : " + str(match))

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]


        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)


        # Draw a label with a name below the face
        # cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        # cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


        # Draw a label with a name above the face
        cv2.rectangle(frame, (left, top - 15), (right, top + 15), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 5, top + 3), font, 0.5, (255, 255, 255), 1)


    cv2.imshow(WINDOW_NAME, frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


    # Write the resulting image to the output video file
    if shouldRecord == True:
    	out.write(frame)



# All done!
video_capture.release()
if shouldRecord == True:
	out.release()
cv2.destroyAllWindows()





