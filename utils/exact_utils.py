import uuid
import urllib3
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from omegaconf import ListConfig

from exact_sync.v1.api.annotations_api import AnnotationsApi
from exact_sync.v1.api.annotation_types_api import AnnotationTypesApi
from exact_sync.v1.api.image_sets_api import ImageSetsApi
from exact_sync.v1.api.images_api import ImagesApi
from exact_sync.v1.api.teams_api import TeamsApi
from exact_sync.v1.api.products_api import ProductsApi

from exact_sync.v1.api_client import ApiClient
from exact_sync.v1.configuration import Configuration
from exact_sync.v1.models import Image, Annotation



class ExactHandle:
    def __init__(self, host, user, pw):
        self.config = Configuration()
        self.config.verify_ssl = False
        self.config.host = host
        self.config.username = user
        self.config.password = pw

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.client = ApiClient(self.config)

        self.images_api = ImagesApi(self.client)
        self.image_sets_api = ImageSetsApi(self.client)
        self.annotations_api = AnnotationsApi(self.client)
        self.annotation_types_api = AnnotationTypesApi(self.client)
        self.teams_api = TeamsApi(self.client)
        self.products_api = ProductsApi(self.client)


    def get_images(self, imageset, wsi_folder):
        # get image_set(s)
        exact_imagesets = self.get_imagesets(imageset)

        # calculate total number of contained images
        total_img_num = sum([len(image_list.images) for image_list in exact_imagesets])

        # download images and return image_id, path, image_name
        images = []
        with tqdm(total=total_img_num, desc="Downloading images") as pbar:
            for exact_imageset in exact_imagesets:
                exact_images = self.images_api.list_images(image_set=exact_imageset.id, limit=5000).results
                for image in exact_images:
                    image_path = Path(wsi_folder)/image.name

                    # if image filed does not exist, download it
                    if not image_path.is_file():
                        self.images_api.download_image(id=image.id, target_path=image_path, original_image=False)

                    images.append((image.id, image_path, image.name))
                    pbar.update(1)
        
        return images

        
    def get_annotations(self, image_list, imageset, user=None, max_requests=50000):
        # create threads to load each images annotations asynchronously
        thread_list = []
        for image in tqdm(image_list, desc="Requesting annotations"):
            if user is None:
                annos = self.annotations_api.list_annotations(async_req=True, image=image[0], deleted=False, pagination=True, limit=max_requests)
            else:
                annos = self.annotations_api.list_annotations(async_req=True, image=image[0], deleted=False, pagination=True, limit=max_requests, user=user)

            thread_list.append((annos, image[0], image[1]))

        # get current imagesets
        exact_imagesets = self.get_imagesets(imageset)
        # create lookup for product names
        product_dict = {}
        for exact_imageset in exact_imagesets:
            exact_products = self.products_api.list_products(imagesets=exact_imageset.id).results
            for exact_product in exact_products:
                product_dict[exact_product.id] = exact_product.name

        # create lookup for annotation names and their product
        annotation_names = {}
        product_names = {}
        for product_id in product_dict.keys():
            for anno_type in self.annotation_types_api.list_annotation_types(product=product_id).results:
                annotation_names[anno_type.id] = anno_type.name
                product_names[anno_type.id] = product_dict[product_id]

        # collect threads and process annotations
        annotations = []
        for anno_tuple in tqdm(thread_list, desc="Collecting annotations"):
            annos = anno_tuple[0].get().results

            if len(annos) == max_requests:
                raise Exception(f"Max annotation request limit of {max_requests} not sufficient.")  
                
            # get the relevant informations from the annotation
            for anno in annos:
                anno_name = annotation_names[anno.annotation_type]
                product_name = product_names[anno.annotation_type]
                annotations.append([anno_tuple[1], anno.vector, anno_name, product_name, anno.id, anno.unique_identifier, anno.last_edit_time])

        # create pandas datatframe for easier handling
        return pd.DataFrame(annotations, columns=["Image", "Vector", "Label", "Product", "ID", "UUID", "Time"])


    def get_imagesets(self, imageset):
        if type(imageset) == str:
            exact_imagesets = self.image_sets_api.list_image_sets(name=imageset).results
        elif type(imageset) in [list,tuple, ListConfig]:
            exact_imagesets = []
            for imgset_name in imageset:
                exact_imagesets.extend(self.image_sets_api.list_image_sets(name=imgset_name).results)
        else:
            raise Exception("Unkown imageset format. Use a single string or a list/tuple of strings.")

        return exact_imagesets


    def upload_image(self, image, imageset):
        if type(image) in [list,tuple]:
            imgs_to_upload = image
        else:
            imgs_to_upload = [image]

        # get imageset
        exact_imageset = self.image_sets_api.list_image_sets(name=imageset).results[0]

        # upload images if not already existing
        for img_to_upload in tqdm(imgs_to_upload, desc="Uploading images"):
            img_path = Path(img_to_upload)
            ret = self.images_api.list_images(name=img_path.name, image_set=exact_imageset.id, limit=5000)
        
            if ret.count == 0:
                image_type = int(Image.ImageSourceTypes.DEFAULT)
                self.images_api.create_image(file_path=img_to_upload, image_type=image_type, image_set=exact_imageset.id)

    
    def clear_all_annotations(self, imageset, images=None, max_requests=50000, clear_chunk_size=20):
        # get imageset
        exact_imageset = self.image_sets_api.list_image_sets(name=imageset).results[0]

        # get images to clear
        if images is None:
            exact_img_list = self.images_api.list_images(image_set=exact_imageset.id, limit=5000).results
        else:
            exact_img_list = []
            for image in images:
                exact_img = self.images_api.list_images(image_set=exact_imageset.id, name=image, limit=5000).results
                exact_img_list.extend(exact_img)
                
        # collect annotations from each image
        clear_list = []
        for exact_image in tqdm(exact_img_list, desc="Collecting annotations to clear"):
            annos = self.annotations_api.list_annotations(image=exact_image.id, deleted=False, limit=max_requests).results

            if len(annos) == max_requests:
                raise Exception(f"Max annotation request limit of {max_requests} not sufficient.")

            clear_list.extend([str(elem.id) for elem in annos])

        # clear all annotations
        clear_thread_list = []
        for i in tqdm(range(0, len(clear_list), clear_chunk_size), desc="Creating async clear requests"):
            clear_string = ','.join(clear_list[i:min(i+clear_chunk_size, len(clear_list))])
            clear_thread_list.append(self.annotations_api.multiple_delete(clear_string, async_req=True))

        for clear_thread in tqdm(clear_thread_list, "Collecting async clear requests"):
            clear_thread.get()


    def upload_annotations(self, annotation_list, imageset, product): # annotation_list: [(Label (str), Vector (dict/2d numpy), Image (str)),]
        # get imageset
        exact_imageset = self.image_sets_api.list_image_sets(name=imageset).results[0]
        # get product
        exact_product = self.products_api.list_products(name=product).results[0]

        # create lookup for image names
        img_names = {}
        for exact_img in self.images_api.list_images(image_set=exact_imageset.id, limit=5000).results:
            img_names[exact_img.name] = exact_img.id

        # create lookup for annotation names
        anno_types = {}
        for anno_type in self.annotation_types_api.list_annotation_types(product=exact_product.id).results:
            anno_types[anno_type.name] = anno_type.id

        # upload annotations
        upload_thread_list = []
        for i in tqdm(range(0, len(annotation_list), 100), desc="Creating async upload requests"):
            anno_to_upload_list = annotation_list[i:min(i+100, len(annotation_list))]
            annos = []
            for anno_to_upload in anno_to_upload_list:
                anno_type = anno_types[anno_to_upload[0]]
                anno_vector = anno_to_upload[1]
                anno_image = img_names[anno_to_upload[2]]
                unique_identifier = str(uuid.uuid4())

                exact_anno = Annotation(annotation_type=anno_type, vector=anno_vector, image=anno_image, unique_identifier=unique_identifier)
                annos.append(exact_anno)
            
            upload_thread = self.annotations_api.create_annotation(body=annos, async_req=True)
            upload_thread_list.append(upload_thread)
        
        for upload_thread in tqdm(upload_thread_list, desc="Collecting async upload requests"):
            upload_thread.get()

