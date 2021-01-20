import scipy.ndimage as ndimage
import skimage.measure
import numpy as np
from torch.utils.data import Dataset
import os
import sys
import SimpleITK as sitk
import pydicom as pyd
import logging
from tqdm import tqdm
import fill_voids
import skimage.morphology
import os
import errno
import math
import shutil
import logging
import sys


def preprocess(img, label=None, resolution=[192, 192]):
    imgmtx = np.copy(img)
    lblsmtx = np.copy(label)

    imgmtx[imgmtx < -1024] = -1024
    imgmtx[imgmtx > 600] = 600
    cip_xnew = []
    cip_box = []
    cip_mask = []
    for i in range(imgmtx.shape[0]):
        if label is None:
            (im, m, box) = crop_and_resize(imgmtx[i, :, :], width=resolution[0], height=resolution[1])
        else:
            (im, m, box) = crop_and_resize(imgmtx[i, :, :], mask=lblsmtx[i, :, :], width=resolution[0],
                                           height=resolution[1])
            cip_mask.append(m)
        cip_xnew.append(im)
        cip_box.append(box)
    if label is None:
        return np.asarray(cip_xnew), cip_box
    else:
        return np.asarray(cip_xnew), cip_box, np.asarray(cip_mask)


def get_input_image(path):
    if os.path.isfile(path):
        logging.info(f'Read input: {path}')
        input_image = sitk.ReadImage(path)
    else:
        raise NotImplementedError
    return input_image


def simple_bodymask(img):
    maskthreshold = -500
    oshape = img.shape
    img = ndimage.zoom(img, 128/np.asarray(img.shape), order=0)
    bodymask = img > maskthreshold
    bodymask = ndimage.binary_closing(bodymask)
    bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((3, 3))).astype(int)
    bodymask = ndimage.binary_erosion(bodymask, iterations=2)
    bodymask = skimage.measure.label(bodymask.astype(int), connectivity=1)
    regions = skimage.measure.regionprops(bodymask.astype(int))
    if len(regions) > 0:
        max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1
        bodymask = bodymask == max_region
        bodymask = ndimage.binary_dilation(bodymask, iterations=2)
    real_scaling = np.asarray(oshape)/128
    return ndimage.zoom(bodymask, real_scaling, order=0)


def crop_and_resize(img, mask=None, width=192, height=192):
    bmask = simple_bodymask(img)
    # img[bmask==0] = -1024 # this line removes background outside of the lung.
    # However, it has been shown problematic with narrow circular field of views that touch the lung.
    # Possibly doing more harm than help
    reg = skimage.measure.regionprops(skimage.measure.label(bmask))
    if len(reg) > 0:
        bbox = np.asarray(reg[0].bbox)
    else:
        bbox = (0, 0, bmask.shape[0], bmask.shape[1])
    img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    img = ndimage.zoom(img, np.asarray([width, height]) / np.asarray(img.shape), order=1)
    if not mask is None:
        mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        mask = ndimage.zoom(mask, np.asarray([width, height]) / np.asarray(mask.shape), order=0)
        # mask = ndimage.binary_closing(mask,iterations=5)
    return img, mask, bbox


## For some reasons skimage.transform leads to edgy mask borders compared to ndimage.zoom
# def reshape_mask(mask, tbox, origsize):
#     res = np.ones(origsize) * 0
#     resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
#     imgres = skimage.transform.resize(mask, resize, order=0, mode='constant', cval=0, anti_aliasing=False, preserve_range=True)
#     res[tbox[0]:tbox[2], tbox[1]:tbox[3]] = imgres
#     return res


def reshape_mask(mask, tbox, origsize):
    res = np.ones(origsize) * 0
    resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
    imgres = ndimage.zoom(mask, resize / np.asarray(mask.shape), order=0)
    res[tbox[0]:tbox[2], tbox[1]:tbox[3]] = imgres
    return res


class LungLabelsDS_inf(Dataset):
    def __init__(self, ds):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx, None, :, :].astype(np.float)


def postrocessing(label_image, spare=[]):
    '''some post-processing mapping small label patches to the neighbout whith which they share the
        largest border. All connected components smaller than min_area will be removed
    '''

    # merge small components to neighbours
    regionmask = skimage.measure.label(label_image)
    origlabels = np.unique(label_image)
    origlabels_maxsub = np.zeros((max(origlabels) + 1,), dtype=np.uint32)  # will hold the largest component for a label
    regions = skimage.measure.regionprops(regionmask, label_image)
    regions.sort(key=lambda x: x.area)
    regionlabels = [x.label for x in regions]

    # will hold mapping from regionlabels to original labels
    region_to_lobemap = np.zeros((len(regionlabels) + 1,), dtype=np.uint8)
    for r in regions:
        if r.area > origlabels_maxsub[r.max_intensity]:
            origlabels_maxsub[r.max_intensity] = r.area
            region_to_lobemap[r.label] = r.max_intensity

    for r in tqdm(regions):
        if (r.area < origlabels_maxsub[r.max_intensity] or r.max_intensity in spare) and r.area>2: # area>2 improves runtime because small areas 1 and 2 voxel will be ignored
            bb = bbox_3D(regionmask == r.label)
            sub = regionmask[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
            dil = ndimage.binary_dilation(sub == r.label)
            neighbours, counts = np.unique(sub[dil], return_counts=True)
            mapto = r.label
            maxmap = 0
            myarea = 0
            for ix, n in enumerate(neighbours):
                if n != 0 and n != r.label and counts[ix] > maxmap and n != spare:
                    maxmap = counts[ix]
                    mapto = n
                    myarea = r.area
            regionmask[regionmask == r.label] = mapto
            # print(str(region_to_lobemap[r.label]) + ' -> ' + str(region_to_lobemap[mapto])) # for debugging
            if regions[regionlabels.index(mapto)].area == origlabels_maxsub[
                regions[regionlabels.index(mapto)].max_intensity]:
                origlabels_maxsub[regions[regionlabels.index(mapto)].max_intensity] += myarea
            regions[regionlabels.index(mapto)].__dict__['_cache']['area'] += myarea

    outmask_mapped = region_to_lobemap[regionmask]
    outmask_mapped[outmask_mapped==spare] = 0 

    if outmask_mapped.shape[0] == 1:
        # holefiller = lambda x: ndimage.morphology.binary_fill_holes(x[0])[None, :, :] # This is bad for slices that show the liver
        holefiller = lambda x: skimage.morphology.area_closing(x[0].astype(int), area_threshold=64)[None, :, :] == 1
    else:
        holefiller = fill_voids.fill

    outmask = np.zeros(outmask_mapped.shape, dtype=np.uint8)
    for i in np.unique(outmask_mapped)[1:]:
        outmask[holefiller(keep_largest_connected_component(outmask_mapped == i))] = i

    return outmask


def bbox_3D(labelmap, margin=2):
    shape = labelmap.shape
    r = np.any(labelmap, axis=(1, 2))
    c = np.any(labelmap, axis=(0, 2))
    z = np.any(labelmap, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    rmin -= margin if rmin >= margin else rmin
    rmax += margin if rmax <= shape[0] - margin else rmax
    cmin, cmax = np.where(c)[0][[0, -1]]
    cmin -= margin if cmin >= margin else cmin
    cmax += margin if cmax <= shape[1] - margin else cmax
    zmin, zmax = np.where(z)[0][[0, -1]]
    zmin -= margin if zmin >= margin else zmin
    zmax += margin if zmax <= shape[2] - margin else zmax
    
    if rmax-rmin == 0:
        rmax = rmin+1

    return np.asarray([rmin, rmax, cmin, cmax, zmin, zmax])


def keep_largest_connected_component(mask):
    mask = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(mask)
    resizes = np.asarray([x.area for x in regions])
    max_region = np.argsort(resizes)[-1] + 1
    mask = mask == max_region
    return mask


def convert_3d_2_flat(in_data_matrix):
    num_voxel = np.prod(in_data_matrix.shape)
    return in_data_matrix.reshape(num_voxel)


def convert_flat_2_3d(in_data_array, im_shape):
    return in_data_array.reshape(im_shape)


def read_file_contents_list(file_name):
    print(f'Reading from file list txt {file_name}', flush=True)
    with open(file_name) as file:
        lines = [line.rstrip('\n') for line in file]
        print(f'Number items: {len(lines)}')
        return lines

def save_file_contents_list(file_name, item_list):
    print(f'Save list to file {file_name}')
    print(f'Number items: {len(item_list)}')
    with open(file_name, 'w') as file:
        for item in item_list:
            file.write(item + '\n')


def get_dice(img1, img2):
    assert img1.shape == img2.shape

    img1 = img1.flatten().astype(float)
    img2 = img2.flatten().astype(float)

    dice_val = 2 * (img1 * img2).sum() / (img1 + img2).sum()

    return dice_val

def get_dice_with_effective_mask(img1, img2, mask):
    assert img1.shape == img2.shape
    assert img1.shape == mask.shape

    mask = mask.flatten().astype(float)
    img1 = img1.flatten().astype(float)
    img2 = img2.flatten().astype(float)

    img1 = img1 * mask
    img2 = img2 * mask

    dice_val = 2 * (img1 * img2).sum() / (img1 + img2).sum()

    return dice_val


def get_range_paral_chunk(total_num_item, chunk_pair):
    num_item_each_chunk = int(math.ceil(float(total_num_item) / float(chunk_pair[1])))
    range_lower = num_item_each_chunk * (chunk_pair[0] - 1)
    # range_upper = num_item_each_chunk * chunk_pair[0] - 1
    range_upper = num_item_each_chunk * chunk_pair[0]
    if range_upper > total_num_item:
        range_upper = total_num_item

    return [range_lower, range_upper]


def get_current_chunk(in_list, chunk_pair):
    chunks_list = get_chunks_list(in_list, chunk_pair[1])
    current_chunk = chunks_list[chunk_pair[0] - 1]
    return current_chunk


def get_chunks_list(in_list, num_chunks):
    return [in_list[i::num_chunks] for i in range(num_chunks)]


def get_nii_filepath_and_filename_list(dataset_root):
    nii_file_path_list = []
    subject_list = os.listdir(dataset_root)
    for i in range(len(subject_list)):
        subj = subject_list[i]
        subj_path = dataset_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            sess_path = subj_path + '/' + sess
            nii_files = os.listdir(sess_path)
            for nii_file in nii_files:
                nii_file_path = sess_path + '/' + nii_file
                nii_file_path_list.append(nii_file_path)
                # nii_file_name_list.append(nii_file)


    return nii_file_path_list


def get_nii_filepath_and_filename_list_flat(dataset_root):
    nii_file_path_list = []
    nii_file_name_list = os.listdir(dataset_root)
    for file_name in nii_file_name_list:
        nii_file_path = os.path.join(dataset_root, file_name)
        nii_file_path_list.append(nii_file_path)

    return nii_file_path_list


def get_nii_filepath_and_filename_list_hierarchy(dataset_root):
    nii_file_path_list = []
    nii_file_name_list = []
    subject_list = os.listdir(dataset_root)
    for i in range(len(subject_list)):
        subj = subject_list[i]
        subj_path = dataset_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            sess_path = subj_path + '/' + sess
            nii_files = os.listdir(sess_path)
            for nii_file in nii_files:
                nii_file_path = sess_path + '/' + nii_file
                nii_file_path_list.append(nii_file_path)
                nii_file_name_list.append(nii_file)

    return nii_file_path_list


def get_dataset_path_list(dataset_root, dataset_type):
    file_path_list = []
    if dataset_type == 'flat':
        file_path_list = get_nii_filepath_and_filename_list_flat(dataset_root)
    elif dataset_type == 'hierarchy':
        file_path_list = get_nii_filepath_and_filename_list(dataset_root)
    else:
        file_path_list = []

    return file_path_list


def resample_spore_nifti(spore_nifti_root, spore_resample_root):
    """
    Resample spore data, using c3d
    :param spore_nifti_root:
    :param spore_resample_root:
    :return:
    """
    spore_nii_file_path_list = []
    spore_nii_file_name_list = []
    subject_list = os.listdir(spore_nifti_root)
    for i in range(len(subject_list)):
        subj = subject_list[i]
        subj_path = spore_nifti_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            sess_path = subj_path + '/' + sess
            nii_files = os.listdir(sess_path)
            for nii_file in nii_files:
                nii_file_path = sess_path + '/' + nii_file
                spore_nii_file_path_list.append(nii_file_path)
                spore_nii_file_name_list.append(nii_file)

    file_count = 1
    for iFile in range(len(spore_nii_file_path_list)):
        # if file_count > 3:
        #     break

        file_path = spore_nii_file_path_list[iFile]
        file_name = spore_nii_file_name_list[iFile]

        output_path = spore_resample_root + '/' + file_name

        print('Read image: ', file_path)

        # command_read_info_str = 'c3d ' + file_path + ' -info-full'
        # os.system(command_read_info_str)

        command_str = 'c3d ' + file_path + ' -resample 256x256x180 -o ' + output_path

        os.system(command_str)

        print('Output file: ', file_name, " {}/{}".format(iFile, len(spore_nii_file_name_list)))
        # command_image_info_str = 'c3d ' + output_path + ' -info-full'
        #
        # os.system(command_image_info_str)

        file_count = file_count + 1


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_image_name_from_path(image_path):
    return os.path.basename(image_path)


def dataset_hierarchy_to_flat(in_folder, out_folder):
    file_path_list = get_nii_filepath_and_filename_list(in_folder)
    for file_idx in range(len(file_path_list)):
        file_path = file_path_list[file_idx]
        print(f'({file_idx}/{len(file_path_list)}), Process image {file_path}.')
        file_name = get_image_name_from_path(file_path)
        out_path = os.path.join(out_folder, file_name)
        if os.path.exists(out_path):
            print(out_path + ' already exist')
        else:
            print('Copy file %s to %s' % (file_path, out_path))
            shutil.copyfile(file_path, out_path)


def get_extension(file_full_path):
    filename, file_extension = os.path.splitext(file_full_path)
    return file_extension


def get_registration_command_non_rigid(registration_method_name,
                                       reg_args_affine,
                                       reg_args_non_rigid,
                                       label_file,
                                       reg_tool_root,
                                       fixed_image,
                                       moving_image,
                                       output_image,
                                       output_mat,
                                       output_trans,
                                       output_affine):
    command_list = []
    actual_output_mat_path = output_mat + '_matrix.txt'

    reg_args = reg_args_non_rigid

    if registration_method_name == 'deformable_deedsBCV':
        linearBCVslow_path = os.path.join(reg_tool_root, 'linearBCVslow')
        deedsBCVslow_path = os.path.join(reg_tool_root, 'deedsBCVslow')
        label_prop_command = ''
        if label_file != '':
            label_prop_command = f'-S {label_file}'
        command_list.append(f'{linearBCVslow_path} -F {fixed_image} -M {moving_image} -O {output_mat}')
        command_list.append(f'{deedsBCVslow_path} {reg_args} -F {fixed_image} -M {moving_image} -O {output_image} -A {actual_output_mat_path} {label_prop_command}')
    elif registration_method_name == 'deformable_deedsBCV_paral':
        linearBCV_path = os.path.join(reg_tool_root, 'linearBCV')
        deedsBCV_path = os.path.join(reg_tool_root, 'deedsBCV')

        label_prop_command = ''
        if label_file != '':
            label_prop_command = f'-S {label_file}'

        command_list.append(f'{linearBCV_path} -F {fixed_image} -M {moving_image} -O {output_mat}')
        command_list.append(f'{deedsBCV_path} {reg_args} -F {fixed_image} -M {moving_image} -O {output_image} -A {actual_output_mat_path} {label_prop_command}')
    elif registration_method_name == 'deformable_niftyreg':
        reg_aladin_path = os.path.join(reg_tool_root, 'reg_aladin')
        reg_f3d_path = os.path.join(reg_tool_root, 'reg_f3d')

        output_mat_real = output_mat.replace('.nii.gz', '.txt')
        output_affine_im = output_affine
        output_non_rigid_trans = output_trans

        command_list.append(
            f'{reg_aladin_path} {reg_args_affine} -ref {fixed_image} -flo {moving_image} -res {output_affine_im} -aff {output_mat_real}'
        )

        # command_list.append(
        #     f'{reg_f3d_path} -voff {reg_args} -maxit 1000 -sx 10 -ref {fixed_image} -flo {moving_image} -aff {output_mat_real} -cpp {output_non_rigid_trans} -res {output_image}'
        # )

        command_list.append(
            f'{reg_f3d_path} {reg_args_non_rigid} -maxit 1000 -ref {fixed_image} -flo {moving_image} -aff {output_mat_real} -cpp {output_non_rigid_trans} -res {output_image}'
        )

    else:
        command_list.append('TODO')

    return command_list


def get_registration_command(registration_method_name, reg_args, label_file, reg_tool_root, fixed_image, moving_image, output_image, output_mat):

    command_list = []
    actual_output_mat_path = output_mat + '_matrix.txt'

    if registration_method_name == 'affine_flirt':
        flirt_path = os.path.join(reg_tool_root, 'flirt')
        command_str = f'{flirt_path} {reg_args} -dof 12 -in {moving_image} -ref {fixed_image} -out {output_image} -omat {output_mat} '
        command_list.append(command_str)
    elif registration_method_name == 'affine_flirt_zhoubing':
        flirt_path = os.path.join(reg_tool_root, 'flirt')
        # 1. Rigid.
        mid_step_rigid_mat = output_mat + "_rigid.txt"
        mid_step_rigid_im = output_mat + "_rigid.nii.gz"
        command_list.append(f'{flirt_path} -v -dof 6 -in {moving_image} -ref {fixed_image} -omat {mid_step_rigid_mat} -out {mid_step_rigid_im} -nosearch')
        # 2. DOF 9 Affine.
        command_list.append(f'{flirt_path} -v -dof 9 -in {moving_image} -ref {fixed_image} -init {mid_step_rigid_mat} -omat {output_mat} -out {output_image} -nosearch')
    elif registration_method_name == 'affine_nifty_reg':
        reg_aladin_path = os.path.join(reg_tool_root, 'reg_aladin')
        output_mat_real = output_mat.replace('.nii.gz', '.txt')
        command_list.append(f'{reg_aladin_path} -ln 5 -ref {fixed_image} -flo {moving_image} -res {output_image} -aff {output_mat_real}')
    elif registration_method_name == 'affine_nifty_reg_mask':
        reg_aladin_path = os.path.join(reg_tool_root, 'reg_aladin')
        output_mat_real = output_mat.replace('.nii.gz', '.txt')
        fixed_image_mask = fixed_image.replace('.nii.gz', '_mask.nii.gz')
        moving_image_mask = moving_image.replace('.nii.gz', '_mask.nii.gz')
        command_list.append(f'{reg_aladin_path} -ln 5 -ref {fixed_image} -rmask {fixed_image_mask} -flo {moving_image} -fmask {moving_image_mask} -res {output_image} -aff {output_mat_real}')
    elif registration_method_name == 'rigid_nifty_reg':
        reg_aladin_path = os.path.join(reg_tool_root, 'reg_aladin')
        output_mat_real = output_mat.replace('.nii.gz', '.txt')
        command_list.append(
            f'{reg_aladin_path} -rigOnly -ln 5 -ref {fixed_image} -flo {moving_image} -res {output_image} -aff {output_mat_real}')
    elif registration_method_name == 'affine_deedsBCV':
        linearBCVslow_path = os.path.join(reg_tool_root, 'linearBCVslow')
        applyLinearBCVfloat_path = os.path.join(reg_tool_root, 'applyLinearBCVfloat')
        command_list.append(f'{linearBCVslow_path} -F {fixed_image} -M {moving_image} -O {output_mat}')
        command_list.append(f'{applyLinearBCVfloat_path} -M {moving_image} -A {actual_output_mat_path} -D {output_image}')
    elif registration_method_name == 'deformable_deedsBCV':
        linearBCVslow_path = os.path.join(reg_tool_root, 'linearBCVslow')
        deedsBCVslow_path = os.path.join(reg_tool_root, 'deedsBCVslow')

        label_prop_command = ''
        if label_file != '':
            label_prop_command = f'-S {label_file}'

        command_list.append(f'{linearBCVslow_path} -F {fixed_image} -M {moving_image} -O {output_mat}')
        command_list.append(f'{deedsBCVslow_path} {reg_args} -F {fixed_image} -M {moving_image} -O {output_image} -A {actual_output_mat_path} {label_prop_command}')
    elif registration_method_name == 'deformable_deedsBCV_paral':
        linearBCV_path = os.path.join(reg_tool_root, 'linearBCV')
        deedsBCV_path = os.path.join(reg_tool_root, 'deedsBCV')

        label_prop_command = ''
        if label_file != '':
            label_prop_command = f'-S {label_file}'

        command_list.append(f'{linearBCV_path} -F {fixed_image} -M {moving_image} -O {output_mat}')
        command_list.append(f'{deedsBCV_path} {reg_args} -F {fixed_image} -M {moving_image} -O {output_image} -A {actual_output_mat_path} {label_prop_command}')
    elif registration_method_name == 'deformable_niftyreg':
        reg_aladin_path = os.path.join(reg_tool_root, 'reg_aladin')
        reg_f3d_path = os.path.join(reg_tool_root, 'reg_f3d')

        output_mat_real = output_mat.replace('.nii.gz', '.txt')
        output_affine_im = output_image.replace('.nii.gz', '_affine.nii.gz')
        output_non_rigid_trans = output_image.replace('.nii.gz', '_non_rigid_trans.nii.gz')

        command_list.append(
            f'{reg_aladin_path} -ln 5 -omp 32 -ref {fixed_image} -flo {moving_image} -res {output_affine_im} -aff {output_mat_real}'
        )

        command_list.append(
            f'{reg_f3d_path} -ln 5 -omp 32 -maxit 1000 {reg_args} -ref {fixed_image} -flo {moving_image} -aff {output_mat_real} -cpp {output_non_rigid_trans} -res {output_image}'
        )

    else:
        command_list.append('TODO')

    return command_list


def get_interpolation_command(interp_type_name, bash_config, src_root, moving_image):

    command_list = []
    file_name = moving_image
    real_mat_name = file_name.replace('nii.gz', 'txt')

    bash_script_path = ''
    if interp_type_name == 'clipped_ori':
        bash_script_path = os.path.join(src_root, 'tools/interp_clipped_roi.sh')
    elif interp_type_name == 'full_ori':
        bash_script_path = os.path.join(src_root, 'tools/interp_full_ori.sh')
    elif interp_type_name == 'roi_lung_mask':
        bash_script_path = os.path.join(src_root, 'tools/interp_ori_lung_mask.sh')

    command_list.append(f'{bash_script_path} {bash_config} {file_name} {real_mat_name}')

    return command_list


loggers = {}


def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger