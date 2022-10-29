import cv2
from facenet_pytorch import MTCNN
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)


def detect_box(file_path, idx):
    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    out = cv2.VideoWriter(f'output{idx}-box.mp4', fourcc, 25.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(cap.isOpened(), fps)
    frame_count = 0
    errors = []
    while True: # 무한 루프
        ret, frame = cap.read() # 두 개의 값을 반환하므로 두 변수 지정
        if not ret: # 새로운 프레임을 못받아 왔을 때 braek
            break
        faces, _ = mtcnn.detect(frame)
        frame_count += 1

        if faces is not None:
            for face in faces:
                try:
                    s_x, s_y, e_x, e_y = map(int, face)
                    cv2.rectangle(frame, (s_x, s_y), (e_x, e_y),
                                  (0, 255, 0), 2)
                except:
                    errors.append(frame_count/fps)

        out.write(frame)
        # 10ms 기다리고 다음 프레임으로 전환, Esc누르면 while 강제 종료
        if cv2.waitKey(10) == 27:
            break
    print(len(errors))
    out.release()
    cap.release()  # 사용한 자원 해제
    cv2.destroyAllWindows()


def detect_mosaic(file_path, idx):
    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    out = cv2.VideoWriter(f'output{idx}-mosaic.mp4', fourcc, 25.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(cap.isOpened(), fps)
    frame_count = 0
    errors = []
    while True:
        ret, frame = cap.read()  # 두 개의 값을 반환하므로 두 변수 지정
        if not ret:  # 새로운 프레임을 못받아 왔을 때 break
            break
        faces, _ = mtcnn.detect(frame)
        frame_count += 1

        if faces is not None:
            for face in faces:
                try:
                    s_x,s_y,e_x,e_y = map(int, face)
                    face_img = frame[s_y:e_y+1, s_x:e_x+1]  # 인식된 얼굴 이미지 crop
                    w, h = face_img.shape[0], face_img.shape[1]
                    face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.1, fy=0.1) # 축소
                    face_img = cv2.resize(face_img, (h, w), interpolation=cv2.INTER_AREA) # 확대
                    frame[s_y:e_y+1, s_x:e_x+1] = face_img # 인식된 얼굴 영역 모자이크 처리
                except:
                    errors.append(frame_count/fps)

        out.write(frame)
        # 10ms 기다리고 다음 프레임으로 전환, Esc누르면 while 강제 종료
        if cv2.waitKey(10) == 27:
            break

    print(len(errors))
    out.release()
    cap.release() # 사용한 자원 해제
    cv2.destroyAllWindows()
