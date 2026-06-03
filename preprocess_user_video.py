import os, cv2, face_alignment, tempfile, face_alignment, argparse, subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path

class UserVideoProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.input_size = opt.input_size
        self.sampling_rate = opt.sampling_rate

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    def img_loader(self, path:str) -> np.ndarray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def convert_video_into_frames(self, video_path, dst_dir) -> str:
        with open(os.devnull, 'wb') as f:
            command = ("ffmpeg -y -loglevel panic -i %s -qscale:v 0 %s" % \
                                (video_path, os.path.join(dst_dir, '%05d.jpg')))
            subprocess.call(command, shell=True, stdout=f, stderr=f)                 
        return dst_dir
        
    def extract_audio(self, video_path: str, audio_path: str):
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(self.sampling_rate), "-c:a", "pcm_f32le", audio_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return audio_path

    def dectect_and_crop_face_from_frames(self, frame_dir: str, output_dir:str, pad_ratio: float=1.0):
        frame_paths = sorted(Path(frame_dir).rglob("*.jpg"))
        bsys, bsxs, mys, mxs = [], [], [], []

        for frame_path in frame_paths:
            frame_path = str(frame_path)
            frame = self.img_loader(frame_path)
            h, w = frame.shape[0:2]
            mult = 360. / frame.shape[0]

            resized_frame = cv2.resize(frame, dsize=(0, 0), fx = mult, fy = mult, interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)        
            bboxes = self.fa.face_detector.detect_from_image(resized_frame)
            bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score) for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
            bboxes = bboxes[0]

            bsy = int((bboxes[3] - bboxes[1]) / 2)
            bsx = int((bboxes[2] - bboxes[0]) / 2)
            my  = int((bboxes[1] + bboxes[3]) / 2)
            mx  = int((bboxes[0] + bboxes[2]) / 2)
            bsys.append(bsy)
            bsxs.append(bsx)
            mys.append(my)
            mxs.append(mx)

        bsy = np.mean(bsys)
        bsx = np.mean(bsxs)
        mx = np.mean(mxs)
        my = np.mean(mys)

        bs = int(max(bsy, bsx) * (1+pad_ratio))
        x1, y1 = mx - bs, my - bs
        x2, y2 = mx + bs, my + bs
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        bsx, bsy = x2 - x1, y2 - y1
        mx, my = int(x1 + bsx // 2), int(y1 + bsy // 2)
        bs = int(min(bsx, bsy) // 2)

        for frame_path in frame_paths:
            frame_path = str(frame_path)
            frame = cv2.imread(frame_path)
            face = frame[my - bs: my + bs, mx-bs:mx + bs]
            face = cv2.resize(face, dsize=(self.opt.input_size,self.opt.input_size), interpolation = cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_dir, os.path.basename(frame_path)), face)
        return face

    def prepare_user_video(self, video_path: str, output_path: str, pad_ratio: float=1.0) -> None:
        output_path = os.path.join(output_path, os.path.basename(video_path)[:-4])
        os.makedirs(output_path, exist_ok=True)
        self.extract_audio(video_path, output_path + ".wav")

        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_dir = self.convert_video_into_frames(video_path, tmp_dir)
            self.dectect_and_crop_face_from_frames(frame_dir, output_path, pad_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argument for user video preprocess')
    parser.add_argument('--user_video_path', type=str)
    parser.add_argument('--output_path', type=str, default='data')
    parser.add_argument('--pad_ratio', type=float, default=1.0)

    parser.add_argument('--fps', default=25)
    parser.add_argument('--sampling_rate', default=16000)
    parser.add_argument('--input_size', default=512)
    opt = parser.parse_args()

    processor = UserVideoProcessor(opt)
    processor.prepare_user_video(video_path = opt.user_video_path, output_path= opt.output_path, pad_ratio=opt.pad_ratio)
