import cv2
import tqdm
import math
import numpy as np

from skimage.transform import downscale_local_mean
from skimage.measure import block_reduce
from utils.exact_utils import ExactHandle



def wsi_anno_exact(exact_login, img_set_name, dst_wsi_dir, product, anno_file, label_dict, look_f, user=None):
    # create EXACT server handle
    exact_handle = ExactHandle(exact_login[0], exact_login[1], exact_login[2])
    # get images
    images = exact_handle.get_images(img_set_name, dst_wsi_dir)
    # get all annotations
    anno = exact_handle.get_annotations(images, img_set_name, user=user)

    # loop over all images
    list_wsi = []
    for num, img in tqdm.tqdm(enumerate(images), desc="Processing Annotations", total=len(images)):
        # get image specific annotations
        img_annos = anno[anno["Image"] == img[0]]
        path = img[1]

        # get all annos of desirec product
        tissue_annos = img_annos[img_annos["Product"]==product]
        tissue_annos = tissue_annos[tissue_annos["Label"].isin(label_dict.keys())]

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

        # calculate min_x, min_y, max_x, max_y
        min_x, min_y = np.inf, np.inf
        max_x, max_y = -np.inf, -np.inf

        for cont in conts:
            min_vals = np.amin(cont[0], axis=0)
            max_vals = np.amax(cont[0], axis=0)

            min_x = min(min_x, min_vals[0])
            min_y = min(min_y, min_vals[1])
            max_x = max(max_x, max_vals[0])
            max_y = max(max_y, max_vals[1])

        size_x = max_x - min_x
        size_y = max_y - min_y

        # chunks to iterate over
        chunk_size = 16384
        chunks_x = int(math.ceil(size_x / chunk_size))
        chunks_y = int(math.ceil(size_y / chunk_size))

        # create dataset within given hdf5 file
        dset = anno_file.create_dataset(str(num), (chunks_y*chunk_size, chunks_x*chunk_size), chunks=(512, 512), compression="gzip", dtype='uint8')

        # create low res sampling map
        samp_shape = (int((chunks_y*chunk_size)/look_f), int((chunks_x*chunk_size)/look_f))
        samp_chunk_size = (int(chunk_size/look_f), int(chunk_size/look_f))
        sampling_map = np.full(samp_shape, 255, dtype=np.uint8)

        for chunk_y in range(chunks_y):
            for chunk_x in range(chunks_x):
                x_rel = chunk_x*chunk_size
                y_rel = chunk_y*chunk_size

                x_abs = x_rel + min_x
                y_abs = y_rel + min_y

                # create segmentation image chunk
                seg_chunk = np.full((chunk_size, chunk_size), 255, dtype=np.uint8)
                
                for cont in conts:
                    poly = np.copy(cont[0]).reshape((-1, 1, 2)).astype(int)
                    poly[:,:,0] -= x_abs
                    poly[:,:,1] -= y_abs

                    cv2.drawContours(seg_chunk, [poly], -1, cont[2], -1)

                # update sampling map
                samp_chunk = block_reduce(seg_chunk, block_size=look_f, func=np.median, cval=255)

                samp_pos = (chunk_y*samp_chunk_size[0], chunk_x*samp_chunk_size[1])
                sampling_map[samp_pos[0]:samp_pos[0]+samp_chunk_size[0],samp_pos[1]:samp_pos[1]+samp_chunk_size[1]] = samp_chunk

                # remove 255 from segmentation image chunk
                seg_chunk[seg_chunk==255] = 0

                # fill missing borders
                # problem case with 255 borders -> solve somehow? maybe move down after downsample
                seg_chunk = cv2.morphologyEx(seg_chunk, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))

                # save segmentation image chunk into hdf5 file
                dset[y_rel:y_rel+chunk_size,x_rel:x_rel+chunk_size] = seg_chunk

        # compress sampling map
        img_samp = cv2.imencode(".png", sampling_map)[1]

        list_wsi.append((str(path.name), (min_y, min_x), img_samp, num))

    return list_wsi


def wsi_anno_exact_multi(exact_login, img_set_name, dst_wsi_dir, product, anno_file, label_dict, look_f, users):
    if len(users) == 1:
        return wsi_anno_exact(exact_login, img_set_name, dst_wsi_dir, product, anno_file, label_dict, look_f, users[0])
    else:
        Exception("Multi-Annotator for WSI not implemented yet!")


def sample(slide_obj, anno_dset, offset, p_size, b_scale, pos, trans):
    p_h = int((p_size/2)*b_scale)
    seg_crop = anno_dset[pos[0]-p_h:pos[0]+p_h, pos[1]-p_h:pos[1]+p_h]

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