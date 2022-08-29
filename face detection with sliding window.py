import numpy as np
import cyvlfeat as vlfeat
from utils import *
import os.path as osp
from glob import glob
from random import shuffle
from IPython.core.debugger import set_trace
from sklearn.svm import LinearSVC


# '''Extra credit'''
#importing required libraries
'''Extra credit'''

from itertools import product
from math import floor, pi
import numpy as np
import cv2  # opencv 2
from PIL import Image

def findHOGFeatures(I,cell_size):
    n_divs= cell_size
    n_bins=6
    #I = cv2.imread(file)
    if(I.shape[0]<I.shape[1]):
        dims = 2+I.shape[0]
    elif(I.shape[1]<I.shape[0]):
        dims = 2+I.shape[1]
    else:
        dims = 2+I.shape[0]
        
    imge = cv2.resize(I, dsize=(dims, dims), interpolation=cv2.INTER_CUBIC)
    img = np.asarray(imge)
#     plt.imshow(img)
    # Size of HOG vector
    n_HOG = n_divs * n_divs * n_bins

    # Initialize output HOG vector
    # HOG = [0.0]*n_HOG
    HOG = np.zeros((n_HOG, 1))
    # Apply sobel on image to find x and y orientations of the image
    Icv = img #.getNumpyCv2()
    Ix = cv2.Sobel(Icv,cv2.CV_64F,1,0,ksize=3) 
    Iy = cv2.Sobel(Icv,cv2.CV_64F,0,1,ksize=3)

    Ix = Ix.transpose(1, 0, 2)
    Iy = Iy.transpose(1, 0, 2)

    width = img.shape[0]
    height = img.shape[1]

    cellx = round(width / n_divs)  # width of each cell(division)
    celly = round(height / n_divs)  # height of each cell(division)

    #Area of image
    img_area = height * width

    #Range of each bin
    BIN_RANGE = (2 * pi) / n_bins

    # m = 0
    angles = np.arctan2(Iy, Ix)
    magnit = ((Ix ** 2) + (Iy ** 2)) ** 0.5
    it = product(range(n_divs), range(n_divs), range(cellx), range(celly))
    for m, n, i, j in it:
        # grad value
        grad = magnit[m * cellx + i, n * celly + j][0]
        # normalized grad value
        norm_grad = grad / img_area
        # Orientation Angle
        angle = angles[m*cellx + i, n*celly+j][0]
        # (-pi,pi) to (0, 2*pi)
        if angle < 0:
            angle += 2 * pi
        nth_bin = floor(float(angle/BIN_RANGE))
        HOG[((m * n_divs + n) * n_bins + int(nth_bin))] += norm_grad
    return HOG.transpose()

def get_positive_features(train_path_pos, feature_params):
    """
    This function should return all positive training examples (faces) from
    36x36 images in 'train_path_pos'. Each face should be converted into a
    HoG template according to 'feature_params'.
    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features
    Args:
    -   train_path_pos: (string) This directory contains 36x36 face images
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slower because the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time
            (although you don't have to make the detector step size equal a
            single HoG cell).
    Returns:
    -   feats: N x D matrix where N is the number of faces and D is the template
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    positive_files = glob(osp.join(train_path_pos, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################
    
    #declaring variables
    n_cell = np.ceil(win_size/cell_size).astype('int')
    N = len(positive_files*2)
    D = ((win_size//cell_size)**2)*31

    '''#for my hog descriptor feats=[]
    #feats = []'''
    feats = np.random.rand(N, n_cell*n_cell*31)
    
    i = 0
    #nested for loop for taking faces' gradient
    for image in positive_files:
        
        '''#cv2.imread for my loading image in hog descripor
        #im = cv2.imread(image)     
#extracting hog features from my hog descripor
#         hog_feature = findHOGFeatures(im, cell_size)
#         hog_feature = np.array(hog_feature)
#         feats.append(hog_feature[0])
#     feats = np.array(feats)'''
        
        im = load_image_gray(image)
         #extracting hog feature traditionally
        hog_feature = vlfeat.hog.hog(im, cell_size)
        feats[i,:] = np.reshape(hog_feature,(1, D))
        i = i+1

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    
    return feats

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    """
    This function should return negative training examples (non-faces) from any
    images in 'non_face_scn_path'. Images should be loaded in grayscale because
    the positive training data is only available in grayscale (use
    load_image_gray()).
    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features
    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   num_samples: number of negatives to be mined. It is not important for
            the function to find exactly 'num_samples' non-face features. For
            example, you might try to sample some number from each image, but
            some images might be too small to find enough.
    Returns:
    -   N x D matrix where N is the number of non-faces and D is the feature
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################
    D = ((win_size//cell_size)**2)*31
    n_cell = np.ceil(win_size/cell_size).astype('int')
    
    #declaring feature array
    feats = np.random.rand(num_samples * 10, n_cell * n_cell * 31)
    
    '''#feature array for my hog descriptor
    #feats = []'''
    k = 0

    #nested loop for generative negetive hog features
    for file in negative_files:
        
        #cv2 for reading for my hog feature
        #image = cv2.imread(file)
        
        image = load_image_gray(file)

        #image reshaing factor
        scale_factor_list = [1.0, 0.8, 0.65]

        #nested for loop for generating image array of different scale
        for scale_factor in scale_factor_list:
            reshape_img = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor)

            #taking each image dimention
            dim_x = reshape_img.shape[0]
            dim_y = reshape_img.shape[1]
            r = reshape_img

            a = int(reshape_img.shape[0]/36)
            b = int(reshape_img.shape[1]/36)

            for i in range(a):
                for j in range(b):
                    if (i+1)*36 < dim_x and (j+1)*36 < dim_y:
                        s = r[36*i:36*(i+1) , 36*j: 36*(j+1)] 
                        #traditional way of taking hog features.
                        hog_feature = vlfeat.hog.hog(s, cell_size)
                        feats[k,:] = np.reshape(hog_feature,(1, D))
                        k = k+1
           #using my hog descriptor for taking features
#                         hog_feature = findHOGFeatures(s, cell_size)
#                         hog_feature = np.array(hog_feature)
#                         feats.append(hog_feature[0])
#     feats = np.array(feats)


    #off this when running my hog descriptor
    indices = np.random.randint(0,k,13000)
    feats = feats[indices,:]
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return feats

def train_classifier(features_pos, features_neg, c):
    """
    This function trains a linear SVM classifier on the positive and negative
    features obtained from the previous steps. We fit a model to the features
    and return the svm object.
    Args:
    -   features_pos: N X D array. This contains an array of positive features
            extracted from get_positive_feats().
    -   features_neg: M X D array. This contains an array of negative features
            extracted from get_negative_feats().
    Returns:
    -   svm: LinearSVC object. This returns a SVM classifier object trained
            on the positive and negative features.
    """
    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    # svm = PseudoSVM(10,features_pos.shape[1])
    
    #assigning X_train and y_train value for SVM
    y_pos =  np.ones(((features_pos.shape[0]),1))
    y_neg = np.ones(((features_neg.shape[0],1)))*-1
    y_train = np.vstack((y_pos,y_neg))
    X_train = np.vstack((features_pos,features_neg))
    
    #declaring SVM
    SVM = LinearSVC(penalty = 'l2', loss = 'squared_hinge', dual = True, tol = 1e-4, C =c, multi_class ='ovr', fit_intercept = True,
                    intercept_scaling=1, class_weight=None, verbose=0, random_state=0,  
                      max_iter = 2000)
    svm = SVM.fit(X_train, y_train)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return svm

def mine_hard_negs(non_face_scn_path, svm, feature_params):
    """
    This function is pretty similar to get_random_negative_features(). The only
    difference is that instead of returning all the extracted features, you only
    return the features with false-positive prediction.
    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features
    -   svm.predict(feat): predict features
    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   svm: LinearSVC object
    Returns:
    -   N x D matrix where N is the number of non-faces which are
            false-positive and D is the feature dimensionality.
    """

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################
    
    #assigning variables
    D = ((win_size // cell_size) ** 2) * 31
    N = len(negative_files)
    n_cell = np.ceil(win_size/cell_size).astype('int')
    
    #feats for my hog descriptor
#     features=[]
#     feats = []
    
    feats = np.random.rand(N, n_cell*n_cell*31)
    i = 0
    
    #for loop for getting features
    for file in negative_files:
        #for my hog descriptor
#         #image = cv2.imread(file) 
#         hog_feature = findHOGFeatures(image, cell_size)
#         hog_feature = np.array(hog_feature)
#         feats.append(hog_feature[0])
        
        #taking hog features of all negetive image
        image = load_image_gray(file)
        image = cv2.resize(image, (36, 36))
        hog_feature = vlfeat.hog.hog(image, cell_size)
        features = np.reshape(hog_feature, (1, D))
        y_pred = svm.predict(features)
        
        if(y_pred == 1):
            feats[i,:] = features
            i = i + 1

    
    #for my hog descriptor
    #feats = np.array(feats)
    
    feats = feats[:i-1,:]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats

def run_detector(test_scn_path, svm, feature_params, verbose=False):
    """
    This function returns detections on all of the images in a given path. You
    will want to use non-maximum suppression on your detections or your
    performance will be poor (the evaluation counts a duplicate detection as
    wrong). The non-maximum suppression is done on a per-image basis. The
    starter code includes a call to a provided non-max suppression function.
    The placeholder version of this code will return random bounding boxes in
    each test image. It will even do non-maximum suppression on the random
    bounding boxes to give you an example of how to call the function.
    Your actual code should convert each test image to HoG feature space with
    a _single_ call to vlfeat.hog.hog() for each scale. Then step over the HoG
    cells, taking groups of cells that are the same size as your learned
    template, and classifying them. If the classification is above some
    confidence, keep the detection and then pass all the detections for an
    image to non-maximum suppression. For your initial debugging, you can
    operate only at a single scale and you can skip calling non-maximum
    suppression. Err on the side of having a low confidence threshold (even
    less than zero) to achieve high enough recall.
    Args:
    -   test_scn_path: (string) This directory contains images which may or
            may not have faces in them. This function should work for the
            MIT+CMU test set but also for any other images (e.g. class photos).
    -   svm: A trained sklearn.svm.LinearSVC object
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slowerbecause the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time.
    -   verbose: prints out debug information if True
    Returns:
    -   bboxes: N x 4 numpy array. N is the number of detections.
            bboxes(i,:) is [x_min, y_min, x_max, y_max] for detection i.
    -   confidences: (N, ) size numpy array. confidences(i) is the real-valued
            confidence of detection i.
    -   image_ids: List with N elements. image_ids[i] is the image file name
            for detection i. (not the full path, just 'albert.jpg')
    """
    im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))
    bboxes = np.empty((0, 4))
    confidences = np.empty(0)
    image_ids = []

    # number of top detections to feed to NMS
    topk = 1000 #15

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    # scale_factor = feature_params.get('scale_factor', 0.65)
    template_size = int(win_size / cell_size)


    for idx, im_filename in enumerate(im_filenames):
        print('Detecting faces in {:s}'.format(im_filename))
        im = load_image_gray(im_filename)
        im_id = osp.split(im_filename)[-1]
        im_shape = im.shape
        # create scale space HOG pyramid and return scores for prediction

        #######################################################################
        #                        TODO: YOUR CODE HERE                         #
        #######################################################################

        #assigning threshold value
        k = 0
        threshold = -1

        cur_bboxes_old = np.empty((0, 4))
        cur_confidences_old = np.empty((0))
        
        #assigning scale factors
        scale_factor_list = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

        #nested for loop for getting 
        for scale_factor in scale_factor_list:
            #default variables
#             cur_x_min = (np.random.rand(15,1) * im_shape[1]).astype('int')
#             cur_y_min = (np.random.rand(15,1) * im_shape[0]).astype('int')
#             cur_bboxes = np.hstack([cur_x_min, cur_y_min, \
#                             (cur_x_min + np.random.rand(15,1)*50).astype('int'), \
#                              (cur_y_min + np.random.rand(15,1)*50).astype('int')])
#             cur_confidences = np.random.rand(15)*4 - 2
            #assigning new variables
            cur_confidences = np.random.rand(15)*4 - 2
            cur_x_min = np.empty((0, 1))
            cur_y_min = np.empty((0, 1))
            cur_bboxes = np.empty((0, 4))
            cur_confidences = np.empty((0))

            #getting hog feature for all the resized image
            img = cv2.resize(im, dsize = None, fx = scale_factor, fy = scale_factor)
            hog_features = vlfeat.hog.hog(img, cell_size)
            dim_x =hog_features.shape[0]
            dim_y= hog_features.shape[1]
            
            #for loop for generating sliding window value
            for i in range(dim_x):
                for j in range(dim_y):
                    x_min = i
                    y_min = j
                    if x_min+template_size < dim_x and y_min+template_size < dim_y:
                        feature_window = hog_features[x_min:x_min+template_size,y_min:y_min+template_size]
                        feature_window_reshaped = np.reshape(feature_window,(1, ((win_size // cell_size) ** 2) * 31))
                        confidence = svm.decision_function(feature_window_reshaped)

                        if confidence >= threshold:
                            k = k + 1
                            cur_x_min = np.vstack((cur_x_min,x_min))
                            cur_y_min = np.vstack((cur_y_min,y_min))
                            cur_confidences = np.hstack((cur_confidences,confidence))

            x_min_image =  (cur_x_min*template_size/scale_factor).astype('int')
            y_min_image =  (cur_y_min*template_size/scale_factor).astype('int')
            x_max_image = ((cur_x_min + template_size)*template_size/scale_factor).astype('int')
            y_max_image = ((cur_y_min + template_size)*template_size/scale_factor).astype('int')

            cur_bboxes = np.hstack([y_min_image,x_min_image,y_max_image,x_max_image])
            cur_bboxes_old = np.vstack((cur_bboxes_old,cur_bboxes))
            cur_confidences_old = np.hstack((cur_confidences_old,cur_confidences))

        cur_confidences = cur_confidences_old
        cur_bboxes = cur_bboxes_old

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

        ### non-maximum suppression ###
        # non_max_supr_bbox() can actually get somewhat slow with thousands of
        # initial detections. You could pre-filter the detections by confidence,
        # e.g. a detection with confidence -1.1 will probably never be
        # meaningful. You probably _don't_ want to threshold at 0.0, though. You
        # can get higher recall with a lower threshold. You should not modify
        # anything in non_max_supr_bbox(). If you want to try your own NMS methods,
        # please create another function.


        idsort = np.argsort(-cur_confidences)[:topk]
        cur_bboxes = cur_bboxes[idsort]
        cur_confidences = cur_confidences[idsort]

        is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
            im_shape, verbose=verbose)

        if(len(is_valid_bbox)):
            print('NMS done, {:d} detections passed'.format(sum(is_valid_bbox)))
            cur_bboxes = cur_bboxes[is_valid_bbox]

            cur_confidences = cur_confidences[is_valid_bbox]

            bboxes = np.vstack((bboxes, cur_bboxes))
        confidences = np.hstack((confidences, cur_confidences))
        image_ids.extend([im_id] * len(cur_confidences))


    return bboxes, confidences, image_ids