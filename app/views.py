import os
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def upload_video(request):
    if request.method == 'POST' and request.FILES['video']:
        video_file = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        uploaded_file_url = fs.url(filename)
        
        # Pass the uploaded_file_url to the processing view
        return render(request, 'process_video.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'upload_video.html')

def process_video(request):
    uploaded_file_url = request.GET.get('uploaded_file_url', None)

    if uploaded_file_url:
        # Construct the path to the uploaded file
        uploaded_file_path = os.path.join(settings.MEDIA_URL, uploaded_file_url)
        # print(uploaded_file_path)
        # Initialize OpenCV video capture
        cap = cv2.VideoCapture(str(settings.BASE_DIR) + "/" + uploaded_file_path)
        # print(cap.read()[0])
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        # Initialize MediaPipe pose detection
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # Initialize video writer for output
            processed_video_path = os.path.join(settings.MEDIA_ROOT, 'processed_video1.mp4')
            out = cv2.VideoWriter(processed_video_path,
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))
            # out = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                # print("Frame --",ret)
                if not ret:
                    break
                
                current_frame += 1
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make detection
                results = pose.process(image)
                
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                
                # Write processed frame to output video
                # print(cv2.imshow(image))
                out.write(image)
                # print(image)
                # Optional: Show progress
                progress = int((current_frame / frame_count) * 100)
                # print(f'Processing Progress: {progress}%')
            
            cap.release()
            out.release()
            
            # Provide link to download processed video
            processed_video_path = os.path.join(settings.MEDIA_URL, 'processed_video1.mp4')
            return render(request, 'download_video.html', {
                'processed_file_url': processed_video_path
            })
    
    return HttpResponse('No video file specified.')
