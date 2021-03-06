import os.path
import logging
import time
from collections import OrderedDict
import torch

from utils import utils_logger
from utils import utils_image as util
from model.SRResNet import MSRResNet


def main(model=None, model_path=None):

    utils_logger.logger_info('AIM-track', log_path=os.path.join(model_path, 'AIM-track.log'))
    logger = logging.getLogger('AIM-track')

    # --------------------------------
    # basic settings
    # --------------------------------
    testsets = 'DIV2K'  # DIV2K root path
    testset_L = 'DIV2K_test_LR_bicubic'  # test image folder name

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------
    # load model
    # --------------------------------
    # model_path = os.path.join('MSRResNetx4_model', 'MSRResNetx4.pth')
    if model is None:
        model_path = 'MSRResNetx4_model'
        model = MSRResNet(in_nc=3, out_nc=3, nf=64, nb=16, upscale=4)
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')), strict=True)
    model.eval()
    """ for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device) """

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))
    print('Params number: {}'.format(number_parameters))

    # --------------------------------
    # read image
    # --------------------------------
    L_folder = os.path.join(testsets, testset_L)
    assert os.path.isdir(L_folder)  # check the test images path
    E_folder = os.path.join(model_path, 'results')
    util.mkdir(E_folder)

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info(L_folder)
    logger.info(E_folder)
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for img in util.get_image_paths(L_folder):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        start.record()
        img_E = model(img_L)
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds


#        torch.cuda.synchronize()
#        start = time.time()
#        img_E = model(img_L)
#        torch.cuda.synchronize()
#        end = time.time()
#        test_results['runtime'].append(end-start)  # seconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)

        util.imsave(img_E, os.path.join(E_folder, img_name+ext))
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))
    print('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    model_path = 'logs800/ours_model'  # saved model file path
    from model.SRResNet_Recuv9 import MSRResNet as Model
    model = Model(in_nc=3, out_nc=3, nf=64, nb=3, upscale=4)
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
    model.cuda()
    main(model, model_path)

    # from train import test
    # test(model, epoch=0)
