import re
import cv2
import tqdm
import numpy as np

from numba import njit
from skimage.transform import downscale_local_mean
from skimage.measure import block_reduce
from utils.exact_utils import ExactHandle




def roi_anno_exact(exact_login, img_set_name, dst_wsi_dir, bbox_labels, product, label_dict, look_f, user=None):
    # create EXACT server handle
    exact_handle = ExactHandle(exact_login[0], exact_login[1], exact_login[2])
    # get images
    images = exact_handle.get_images(img_set_name, dst_wsi_dir)
    # get all annotations
    anno = exact_handle.get_annotations(images, img_set_name, user=user)

    # loop over all images
    list_roi = []
    for img in tqdm.tqdm(images, desc="Processing Annotations", total=len(images)):
        # get image specific annotations
        img_annos = anno[anno["Image"] == img[0]]
        path = img[1]

        # get rois for current img
        rois = img_annos[img_annos['Label'].isin(bbox_labels)]
        if len(rois) > 0:
            # get tissue annotations
            tissue_annos = img_annos[img_annos["Product"]==product]

            # create list of all contours and their value
            conts = []
            for anno_vector, anno_label in zip(tissue_annos["Vector"], tissue_annos["Label"]):
                if len(anno_vector):
                    vector = []
                    for i in range(1, (len(anno_vector) // 2) + 1):
                        vector.append([anno_vector['x' + str(i)], anno_vector['y' + str(i)]])
                    poly = np.array(vector)

                    conts.append((poly, cv2.contourArea(poly), label_dict[anno_label]))

            # sort contours by their size (draw large ones first)
            conts.sort(key=lambda el : -el[1])

            for row in rois.iloc:
                # extract image
                bbox_vec = row["Vector"]

                # create segmentation image
                img_seg = np.zeros((bbox_vec["x2"]-bbox_vec["x1"], bbox_vec["y2"]-bbox_vec["y1"]), np.uint8)
                
                for cont in conts:
                    poly = np.copy(cont[0]).reshape((-1, 1, 2)).astype(int)
                    poly[:,:,0] -= bbox_vec["x1"]
                    poly[:,:,1] -= bbox_vec["y1"]

                    cv2.drawContours(img_seg, [poly], -1, cont[2], -1)

                # create low res sampling map
                sampling_map = block_reduce(img_seg, block_size=look_f, func=np.median)

                # compress segmentation and sampling_map
                img_seg = cv2.imencode(".png", img_seg)[1]
                img_samp = cv2.imencode(".png", sampling_map)[1]

                # add to list
                num = int(re.findall(r'\d+', str(path.name))[0])
                list_roi.append((str(path.name), (bbox_vec["y1"], bbox_vec["x1"]), img_seg, img_samp, num))

    return list_roi


def roi_anno_exact_multi(exact_login, img_set_name, dst_wsi_dir, bbox_labels, product, label_dict, look_f, users):
    if len(users) == 1:
        return roi_anno_exact(exact_login, img_set_name, dst_wsi_dir, bbox_labels, product, label_dict, look_f, users[0])
    else:
        # get images for all required users
        lists_roi = []
        for user in users:
            lists_roi.append(roi_anno_exact(exact_login, img_set_name, dst_wsi_dir, bbox_labels, product, label_dict, look_f, user))

        # define consensus function
        @njit
        def consensus_func(anno_values):
            # first find consensus between BG, DCIS, Tumor 
            mod_values = np.zeros_like(anno_values)
            mod_values[np.logical_and(anno_values>=1, anno_values<=4)] = 1
            mod_values[anno_values==5] = 2

            # get first consesus
            first_consensus = int(np.ceil(np.median(mod_values)))

            # if background check if hard negative consensus
            if first_consensus == 0:
                bg_count = np.sum(anno_values==0)
                hn_count = np.sum(anno_values==6)

                if hn_count >= bg_count:
                    consensus_value = 6
                else:
                    consensus_value = 0

                consensus_strength = bg_count + hn_count
            
            elif first_consensus == 1:
                reg_values = anno_values[np.logical_and(anno_values>=1, anno_values<=4)]
                consensus_value = np.ceil(np.median(reg_values))

                count = np.sum(reg_values == consensus_value)
                consensus_value = int(consensus_value)
                consensus_strength = count

            elif first_consensus == 2:
                count = np.sum(anno_values == 5)
                consensus_value = 5
                consensus_strength = count
            
            return consensus_value, consensus_strength

        @njit
        def compute_consens_stack(consens_stack, out_array):
            for j in range(consens_stack.shape[1]):
                for i in range(consens_stack.shape[2]):
                    out_array[:,j,i] = consensus_func(consens_stack[:,j,i])

            return out_array

        # create consensus output
        list_rois = []
        for el in zip(*lists_roi):
            imgs_seg = []
            for img_tup in el:
                imgs_seg.append(cv2.imdecode(img_tup[2], cv2.IMREAD_ANYDEPTH))

            # create consensus stack
            consens_stack = np.stack(imgs_seg, axis=0)

            consensus_array = np.zeros((2, consens_stack.shape[1], consens_stack.shape[2]), dtype=np.uint8)
            compute_consens_stack(consens_stack, consensus_array)

            # calculate consensus image
            consensus_seg = consensus_array[0]
            consensus_strength = consensus_array[1]

            # calculate conensus sampling map
            s = consensus_seg.shape
            new_shape = (int(s[0]/look_f), int(s[1]/look_f))
            consensus_sampling_map = cv2.resize(consensus_seg, new_shape, interpolation=cv2.INTER_NEAREST)

            # compress segmentation and sampling_map
            consensus_seg = cv2.imencode(".png", consensus_seg)[1]
            consensus_sampling_map = cv2.imencode(".png", consensus_sampling_map)[1]
            consensus_strength = cv2.imencode(".png", consensus_strength)[1]

            list_rois.append((el[0][0], el[0][1], consensus_seg, consensus_sampling_map, consensus_strength, el[4][0]))
        
        return list_rois


def sample(slide_obj, offset, seg_comp, p_size, b_scale, pos, trans):
    p_h = int((p_size/2)*b_scale)
    seg_crop = cv2.imdecode(seg_comp, cv2.IMREAD_ANYDEPTH)[pos[0]-p_h:pos[0]+p_h, pos[1]-p_h:pos[1]+p_h]

    img_crop = wsi_sample(slide_obj, offset, p_size, b_scale, pos)
    
    applied = trans(image=img_crop, mask=seg_crop)
    seg_crop = applied['mask']
    img_crop_s0 = applied['image']

    return img_crop_s0, seg_crop


def wsi_sample(slide_obj, offset, p_size, b_scale, pos):
    t_scale = b_scale

    levels = [int(round(level)) for level in slide_obj.level_downsamples]
    if t_scale in levels:
        lvl = int(np.argwhere(np.array(levels)==t_scale)[0])

        y0 = int(pos[0] + offset[0] - (p_size/2)*t_scale)
        x0 = int(pos[1] + offset[1] - (p_size/2)*t_scale)

        img_crop = np.array(slide_obj.read_region(location=(x0, y0), level=lvl, size=(p_size,p_size)), copy=True)
        img_crop[:,:,:3][img_crop[:,:,3]==0] = 255
        img_crop = img_crop[:, :, :3]

    else:
        for level in reversed(levels):
            if level < t_scale:
                out_lvl = level
                break
        
        lvl = int(np.argwhere(np.array(levels)==out_lvl)[0])
        scale_diff = int(t_scale/out_lvl)

        y0 = int(pos[0] + offset[0] - (p_size/2)*t_scale)
        x0 = int(pos[1] + offset[1] - (p_size/2)*t_scale)

        img_crop = np.array(slide_obj.read_region(location=(x0, y0), level=lvl, size=(int(p_size*scale_diff),int(p_size*scale_diff))))
        img_crop[:,:,:3][img_crop[:,:,3]==0] = 255
        img_crop = img_crop[:, :, :3]

        img_crop = downscale_local_mean(img_crop, (scale_diff, scale_diff, 1)).astype(np.uint8)
        
    return img_crop