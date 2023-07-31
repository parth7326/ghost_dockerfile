import torch
import time
import os
import cv2
import string
import random

from utils.inference.image_processing import crop_face, get_final_image
from utils.inference.video_processing import read_video, get_target, get_final_video, add_audio_from_another_video, \
    face_enhancement
from utils.inference.core import model_inference

from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions
from datetime import datetime


class GHOST:
    def __init__(
            self,
            num_blocks,
            backbone,
            G_path,
            use_sr,
            crop_size,
            similarity_th,
            batch_size,
    ):
        self.num_blocks = num_blocks
        self.backbone = backbone
        self.G_path = G_path
        self.use_sr = use_sr
        self.crop_size = crop_size
        self.similarity_th = similarity_th
        self.batch_size = batch_size
        self.app = None
        self.app = None
        self.G = None
        self.netArc = None
        self.handler = None
        self.model = None

    def video_swap(self, target_video, source_images, target_faces_images=None):
        if target_faces_images is None:
            target_faces_images = []
        # self.init_models()

        cropped_source_images = []
        for source_image in source_images:
            try:
                source_image = crop_face(source_image, self.app, self.crop_size)[0]
                cropped_source_images.append(source_image[:, :, ::-1])
            except TypeError as e:
                print("Bad source images! Ignore image.")

        cropped_target_faces_images = []
        for target_face_image in target_faces_images:
            try:
                target_face_image = crop_face(target_face_image, self.app, self.crop_size)[0]
                cropped_target_faces_images.append(target_face_image)
            except TypeError as e:
                print("Bad target faces images! Ignore image.")

        start_time = time.time()

        name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-' + self.get_random_string(8)
        file_name = 'results/videos/original/' + name + '.mp4'
        out_name = 'results/videos/result/' + name + '.mp4'

        file = open(file_name, "wb")
        file.write(target_video)
        file.close()

        full_frames, fps = read_video(file_name)
        final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(
            full_frames,
            cropped_source_images,
            cropped_target_faces_images if len(cropped_target_faces_images) else get_target(
                full_frames,
                self.app,
                self.crop_size
            ),
            self.netArc,
            self.G,
            self.app,
            False,
            similarity_th=self.similarity_th,
            crop_size=self.crop_size,
            BS=self.batch_size
        )
        if self.use_sr:
            final_frames_list = face_enhancement(final_frames_list, self.model)

        get_final_video(
            final_frames_list,
            crop_frames_list,
            full_frames,
            tfm_array_list,
            out_name,
            fps,
            self.handler
        )

        add_audio_from_another_video(file_name, out_name, "audio")

        print('Total time: ', time.time() - start_time)

        f = open(out_name, "rb")
        return f.read()

    def image_swap(self, target_image, source_images, target_faces_images=None):

        name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-' + self.get_random_string(8)
        file_name = 'results/photos/original/' + name + '.jpg'
        out_name = 'results/photos/result/' + name + '.jpg'

        cv2.imwrite(file_name, target_image)

        if target_faces_images is None:
            target_faces_images = []
        # self.init_models()

        cropped_source_images = []

        for source_image in source_images:
            try:
                source_image = crop_face(source_image, self.app, self.crop_size)[0]
                cropped_source_images.append(source_image[:, :, ::-1])
            except TypeError as e:
                print("Bad source images! Ignore image.")

        cropped_target_faces_images = []

        for target_face_image in target_faces_images:
            try:
                target_face_image = crop_face(target_face_image, self.app, self.crop_size)[0]
                cropped_target_faces_images.append(target_face_image)
            except TypeError as e:
                print("Bad target faces images! Ignore image.")

        start_time = time.time()

        final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(
            [target_image],
            cropped_source_images,
            cropped_target_faces_images if len(cropped_target_faces_images) else get_target(
                [target_image],
                self.app,
                self.crop_size
            ),
            self.netArc,
            self.G,
            self.app,
            False,
            similarity_th=self.similarity_th,
            crop_size=self.crop_size,
            BS=self.batch_size
        )
        if self.use_sr:
            final_frames_list = face_enhancement(final_frames_list, self.model)

        result = get_final_image(final_frames_list, crop_frames_list, full_frames[0], tfm_array_list, self.handler)

        cv2.imwrite(out_name, result)

        print('Total time: ', time.time() - start_time)

        return result

    def init_models(self):
        init_model_start = time.time()
        # model for face cropping
        app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

        # main model for generation
        G = AEI_Net(self.backbone, num_blocks=self.num_blocks, c_id=512)
        G.eval()
        G.load_state_dict(torch.load(self.G_path, map_location=torch.device('cpu')))
        G = G.cuda()
        G = G.half()

        # arcface model to get face embedding
        netArc = iresnet100(fp16=False)
        netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
        netArc = netArc.cuda()
        netArc.eval()

        # model to get face landmarks
        handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0, det_size=640)

        # model to make superres of face, set use_sr=True if you want to use super resolution or use_sr=False if you don't
        if self.use_sr:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            torch.backends.cudnn.benchmark = True
            opt = TestOptions()
            # opt.which_epoch ='10_7'
            model = Pix2PixModel(opt)
            model.netG.train()
        else:
            model = None

        self.app = app
        self.G = G
        self.netArc = netArc
        self.handler = handler
        self.model = model

        print('Init model time: ', time.time() - init_model_start)

    def get_random_string(self, length):
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str
