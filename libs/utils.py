import cv2
import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET

def draw_rect(im, cords, img_raw, color = None):
    """Draw the rectangle on the image
    
    Parameters
    ----------
    
    im : numpy.ndarray
        numpy image 
    
    cords: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    Returns
    -------
    
    numpy.ndarray
        numpy image with bounding boxes drawn on it
        
    """
    im = im.copy()
    cords = np.array(cords)
    if len(im.shape) == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    
    cords = cords[:,:4]
    cords = cords.reshape(-1,4)
    if not color:
        color = [0,255,0]
    for cord in cords:
        
        pt1, pt2 = (cord[0], cord[1]) , (cord[2], cord[3])
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
    
        im = cv2.rectangle(im, pt1, pt2, color, int(max(im.shape[:2])/400))
        
    img_debug = np.vstack((img_raw, im))
    cv2.imwrite('_debug_img.jpg', img_debug)
    
    
def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])


def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image
    
    Parameters
    ----------
    
    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`
        
    alpha: float
        If the fraction of a bounding box left in the image after being clipped is 
        less than `alpha` the bounding box is dropped. 
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2` 
    
    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
    
    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
    
    delta_area = ((ar_ - bbox_area(bbox))/ar_)
    
    mask = (delta_area < (1 - alpha)).astype(int)
    
    bbox = bbox[mask == 1,:]


    return bbox


def rotate_im(image, angle):
    """Rotate the image.
    
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 
    
    Parameters
    ----------
    
    image : numpy.ndarray
        numpy image
    
    angle : float
        angle by which the image is to be rotated
    
    Returns
    -------
    
    numpy.ndarray
        Rotated Image
    
    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image


def get_corners(bboxes):
    
    """Get corners of bounding boxes
    
    Parameters
    ----------
    
    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
        
    """
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners


def rotate_box(corners,angle,  cx, cy, h, w):
    
    """Rotate the bounding box.
    
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    
    angle : float
        angle by which the image is to be rotated
        
    cx : int
        x coordinate of the center of image (about which the box will be rotated)
        
    cy : int
        y coordinate of the center of image (about which the box will be rotated)
        
    h : int 
        height of the image
        
    w : int 
        width of the image
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated


def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Returns 
    -------
    
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    """
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding
    
    Parameters
    ----------
    
    img : numpy.ndarray
        Image 
    
    inp_dim: tuple(int)
        shape of the reszied image
        
    Returns
    -------
    
    numpy.ndarray:
        Resized image
    
    '''

    inp_dim = (inp_dim, inp_dim)
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h))
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 0)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas


def get_info_bbox(xml_path, label_mapping):
    """
    Extracts bounding box information from an XML file.

    Args:
        xml_path (str): The path to the XML file containing bounding box information.
        label_mapping (dict): A dictionary that maps class names to their corresponding class IDs.

    Returns:
        numpy.ndarray: An array containing bounding box coordinates and class IDs.
    """
    bboxes = []
    # Read the contents of the XML file
    xml_text = ET.parse(xml_path)
    root = xml_text.getroot()
    for obj in root.iter('object'):
        # Get the class name of the object from the XML file
        class_name = obj.find('name').text
        try:
            # Find the class ID from the label_mapping dictionary
            id_class = label_mapping[class_name]
        except KeyError:
            print(f'Not found class name: {class_name}')
            continue
        for bbox in obj.iter('bndbox'):
            # Get the bounding box coordinates from the XML file
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            # Check if the bounding box has valid dimensions
            assert xmax > xmin and ymax > ymin, f"Box size error!: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
            # Add the bounding box information and class ID to the bboxes list
            bboxes.append((xmin, ymin, xmax, ymax, id_class))
    # Convert the bboxes list to a numpy array with dtype float32
    bboxes = np.array(bboxes, dtype=np.float32)
    
    return bboxes

def create_folder(path):
    """
    Create a new folder at the specified path or remove and recreate it if it already exists.

    Args:
        path (str): The path of the folder to be created.

    Returns:
        None
    """
    # Check if the folder already exists
    if os.path.exists(path):
        # If it exists, remove it and its contents
        shutil.rmtree(path)
    
    # Create a new folder at the specified path
    os.mkdir(path)
    
def bndbox2yololine(box, img):
    """
    Convert bounding box coordinates to YOLO format.

    Args:
        box (numpy array): An array representing the bounding box in the format [xmin, ymin, xmax, ymax, class_id].
        img (numpy array): The input image as a numpy array.

    Returns:
        tuple: A tuple representing the bounding box in YOLO format (class_id, xcen, ycen, w_, h_).
    """
    # Unpack the bounding box coordinates and class ID
    xmin, ymin, xmax, ymax, id_class = box
    
    # Get the height and width of the image
    h, w = img.shape[:2]

    # Calculate the center coordinates of the bounding box relative to the image size
    xcen = float((xmin + xmax)) / 2 / w
    ycen = float((ymin + ymax)) / 2 / h

    # Calculate the width and height of the bounding box relative to the image size
    w_ = float((xmax - xmin)) / w
    h_ = float((ymax - ymin)) / h

    # Return the bounding box in YOLO format as a tuple (class_id, xcen, ycen, w_, h_)
    return float(id_class), xcen, ycen, w_, h_

def save_yolo_format(path_save, bboxes, img):
    """
    Save bounding boxes in YOLO format to a file.

    Args:
        path_save (str): The path to save the YOLO format file.
        bboxes (numpy array): An array representing the bounding boxes in the format [xmin, ymin, xmax, ymax, class_id].
        img (numpy array): The input image as a numpy array.
    """
    # Open the output file in write mode
    out_file = open(path_save, 'w')
    
    # Convert and write each bounding box in YOLO format to the file
    for box in list(bboxes):
        # Convert the bounding box coordinates to YOLO format
        class_index, xcen, ycen, w, h = bndbox2yololine(box, img)
        
        # Write the bounding box in YOLO format to the file
        out_file.write("%d %.6f %.6f %.6f %.6f\n" % (class_index, xcen, ycen, w, h))
    
    # Close the output file
    out_file.close()
    
