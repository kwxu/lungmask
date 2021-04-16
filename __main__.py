import sys
import argparse
import logging
from lungmask import mask
from lungmask import utils
import os
import SimpleITK as sitk
from lungmask.utils import read_file_contents_list, save_file_contents_list


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


in_folder = '/nfs/masi/xuk9/SPORE/data/data_flat'
out_mask_folder = '/nfs/masi/xuk9/src/lungmask/SPORE/lung_mask_nii'
file_name = '00000555time20170316.nii.gz'
file_list_txt = '/nfs/masi/xuk9/SPORE/data/vlsp_data_list.txt'

# in_nii = os.path.join(in_folder, file_name)
# out_nii_mask = os.path.join(out_mask_folder, file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-folder', type=str,
                        default=in_folder)
    parser.add_argument('--out-folder', type=str,
                        default=out_mask_folder)
    parser.add_argument('--file-list-txt', type=str,
                        default=file_list_txt)
    parser.add_argument('--failed-list-txt', type=str)
    parser.add_argument('--modeltype', help='Default: unet', type=str, choices=['unet'], default='unet')
    parser.add_argument('--modelname', help="spcifies the trained model, Default: R231", type=str, choices=['R231','LTRCLobes','LTRCLobes_R231','R231CovidWeb'], default='R231')
    parser.add_argument('--cpu', help="Force using the CPU even when a GPU is available, will override batchsize to 1", action='store_true')
    parser.add_argument('--nopostprocess', help="Deactivates postprocessing (removal of unconnected components and hole filling", action='store_true')
    parser.add_argument('--noHU', help="For processing of images that are not encoded in hounsfield units (HU). E.g. png or jpg images from the web. Be aware, results may be substantially worse on these images", action='store_true')
    parser.add_argument('--batchsize', type=int, help="Number of slices processed simultaneously. Lower number requires less memory but may be slower.", default=20)

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)

    batchsize = args.batchsize
    if args.cpu:
        batchsize = 1

    logging.info(f'Load model')

    file_list = read_file_contents_list(args.file_list_txt)
    failed_case_list = []
    for file_idx in range(len(file_list)):
        file_name = file_list[file_idx]
        logging.info(f'Process {file_name}, {file_idx} / {len(file_list)}')

        try:
            in_nii = os.path.join(args.in_folder, file_name)
            out_nii_mask = os.path.join(args.out_folder, file_name)

            input_image = utils.get_input_image(in_nii)
            logging.info(f'Infer lungmask')
            if args.modelname == 'LTRCLobes_R231':
                result = mask.apply_fused(input_image, force_cpu=args.cpu, batch_size=batchsize,
                                          volume_postprocessing=not (args.nopostprocess), noHU=args.noHU)
            else:
                model = mask.get_model(args.modeltype, args.modelname)
                result = mask.apply(input_image, model, force_cpu=args.cpu, batch_size=batchsize,
                                    volume_postprocessing=not (args.nopostprocess), noHU=args.noHU)

            result_out = sitk.GetImageFromArray(result)
            result_out.CopyInformation(input_image)
            logging.info(f'Save result to: {out_nii_mask}')
            # sys.exit(sitk.WriteImage(result_out, out_nii_mask))
            sitk.WriteImage(result_out, out_nii_mask)
        except:
            print(f'Something wrong with {file_name}')
            failed_case_list.append(file_name)

    save_file_contents_list(args.failed_list_txt, failed_case_list)


if __name__ == "__main__":
    print('called as script')
    main()
